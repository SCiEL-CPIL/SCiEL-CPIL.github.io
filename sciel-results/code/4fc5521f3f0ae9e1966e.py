import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10):
    start_time = time.time()
    
    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = 2 * (val - 1.0)
                    
                    grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_loss = float('inf')
    best_x = None
    
    # Try a few initializations
    for init_idx in range(5):
        if time.time() - start_time > time_limit:
            break
            
        if init_idx == 0:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1
        elif init_idx == 1:
            x0 = np.zeros(n*m*r + m*p*r + p*n*r)
            num_nonzero = r * 3
            indices = np.random.choice(len(x0), num_nonzero, replace=False)
            x0[indices] = np.random.choice([-1, 1], num_nonzero)
        else:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
                
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap, V_snap, W_snap, True
                    
        except Exception as e:
            pass
            
    if best_x is not None and best_loss < 1e-6:
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, n*m*p + 1):
        if time.time() > deadline - 5:
            break
            
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=10)
        
        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
