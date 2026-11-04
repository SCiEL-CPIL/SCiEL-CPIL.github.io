import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    if len(U) == 0: return False
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half_int(arr):
    """Snap array values to the nearest half-integer (multiple of 0.5)."""
    return np.round(arr * 2) / 2.0

def _is_half_integer(arr, tol=1e-6):
    """Check if array values are close to half-integers."""
    scaled = arr * 2
    rounded = np.round(scaled)
    return np.all(np.abs(scaled - rounded) < tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Schoolbook solution
    U_school = np.zeros((nm, n*m*p))
    V_school = np.zeros((mp, n*m*p))
    W_school = np.zeros((pn, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n*m*p
    
    # Objective function and gradient for optimization
    def objective_and_grad(x, r):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        
        # Compute S[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        S = np.zeros((n, m, p))
        for c in range(r):
            # Outer product of U_col, V_col, W_col
            # U_col shape (nm,), V_col shape (mp,), W_col shape (pn,)
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            # Reshape to (n, m), (m, p), (p, n)
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            
            # Compute contribution to S
            # S[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is a rank-1 tensor addition
            # We can do this efficiently
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i,j]
                    if u_val == 0: continue
                    for k in range(p):
                        S[i,j,k] += u_val * v_mat[j,k] * w_mat[k,i]
        
        # Loss: sum (S - 1)^2
        error = S - 1.0
        loss = np.sum(error**2)
        
        # Gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(r):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            
            # dLoss/dU[i,j,c] = 2 * sum_k error[i,j,k] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    # Sum over k
                    s_val = 0.0
                    for k in range(p):
                        s_val += error[i,j,k] * v_mat[j,k] * w_mat[k,i]
                    grad_U[i*m+j, c] = 2 * s_val
                    
            # dLoss/dV[j,k,c] = 2 * sum_i error[i,j,k] * u_mat[i,j] * w_mat[k,i]
            for j in range(m):
                for k in range(p):
                    s_val = 0.0
                    for i in range(n):
                        s_val += error[i,j,k] * u_mat[i,j] * w_mat[k,i]
                    grad_V[j*p+k, c] = 2 * s_val
                    
            # dLoss/dW[k,i,c] = 2 * sum_j error[i,j,k] * u_mat[i,j] * v_mat[j,k]
            for k in range(p):
                for i in range(n):
                    s_val = 0.0
                    for j in range(m):
                        s_val += error[i,j,k] * u_mat[i,j] * v_mat[j,k]
                    grad_W[k*n+i, c] = 2 * s_val
                    
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    np.random.seed(42)
    
    # Try ranks from 20 upwards
    for r in range(20, 24):
        if time.time() > deadline - 10:
            break
            
        max_attempts = 30 if r >= 22 else 15
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use small values to help convergence
            U_init = np.random.uniform(-1, 1, (nm, r))
            V_init = np.random.uniform(-1, 1, (mp, r))
            W_init = np.random.uniform(-1, 1, (pn, r))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                res = minimize(
                    objective_and_grad, 
                    x0, 
                    args=(r,), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-8:
                # Check if we found a solution
                x_opt = res.x
                U_opt = x_opt[:nm*r].reshape(nm, r)
                V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
                W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
                
                # Try to snap to half-integers
                U_snap = _snap_to_half_int(U_opt)
                V_snap = _snap_to_half_int(V_opt)
                W_snap = _snap_to_half_int(W_opt)
                
                # Verify snapped solution
                if _verify(U_snap, V_snap, W_snap):
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    best_rank = r
                    found = True
                    break
                else:
                    # If snapped version is invalid, check if the real version is valid 
                    # (in case evaluator is lenient, but prompt says integer/half-integer)
                    # We'll stick to the strict requirement for now.
                    # However, if the real solution is very close to half-integers, 
                    # maybe we can try a local search around the snapped values?
                    pass
        
        if found:
            break

    return best_result
