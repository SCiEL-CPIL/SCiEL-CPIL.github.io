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

def _compute_error(U, V, W, n, m, p):
    """Compute the total squared error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return err

def _local_search_half_integer(U, V, W, n, m, p, max_iterations=500, time_limit=15):
    """
    Local search on half-integer space to minimize decomposition error.
    Tries perturbing individual entries by ±0.5.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    current_err = _compute_error(U, V, W, n, m, p)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try perturbing U
        for row in range(U.shape[0]):
            for col in range(U.shape[1]):
                if time.time() - start_time > time_limit:
                    break
                old_val = U[row, col]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    U[row, col] = new_val
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        U[row, col] = old_val
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break
        
        # Try perturbing V
        for row in range(V.shape[0]):
            for col in range(V.shape[1]):
                if time.time() - start_time > time_limit:
                    break
                old_val = V[row, col]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    V[row, col] = new_val
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        V[row, col] = old_val
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break
        
        # Try perturbing W
        for row in range(W.shape[0]):
            for col in range(W.shape[1]):
                if time.time() - start_time > time_limit:
                    break
                old_val = W[row, col]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    W[row, col] = new_val
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err - 1e-12:
                        current_err = new_err
                        improved = True
                        break
                    else:
                        W[row, col] = old_val
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
        
        if not improved:
            break
    
    return U, V, W, current_err

def _optimize_rank_343(r, time_limit=20):
    """
    Optimizes for rank r decomposition of 3x4x3 matrix multiplication.
    Returns (U, V, W, success)
    """
    n, m, p = 3, 4, 3
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
    
    # Try multiple initializations
    num_inits = 15
    for init_idx in range(num_inits):
        if time.time() - start_time > time_limit - 5:
            break
            
        if init_idx == 0:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1
        elif init_idx == 1:
            # Sparse initialization
            x0 = np.zeros(n*m*r + m*p*r + p*n*r)
            num_nonzero = min(r * 5, len(x0) // 2)
            indices = np.random.choice(len(x0), num_nonzero, replace=False)
            x0[indices] = np.random.choice([-1, -0.5, 0.5, 1], num_nonzero)
        elif init_idx == 2:
            # Near-schoolbook initialization with one term removed
            # Start from schoolbook and zero out one column
            x0 = np.zeros(n*m*r + m*p*r + p*n*r)
            U_part = x0[:n*m*r].reshape(n*m, r)
            V_part = x0[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_part = x0[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Fill with schoolbook-like structure for r terms
            for t in range(r):
                i = t // (m*p)
                rem = t % (m*p)
                j = rem // p
                k = rem % p
                U_part[i*m+j, t] = 1
                V_part[j*p+k, t] = 1
                W_part[k*n+i, t] = 1
            
            # Add small random perturbation
            x0 += np.random.randn(len(x0)) * 0.1
        elif init_idx == 3:
            # Structured initialization with some symmetry
            x0 = np.zeros(n*m*r + m*p*r + p*n*r)
            U_part = x0[:n*m*r].reshape(n*m, r)
            V_part = x0[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_part = x0[n*m*r + m*p*r :].reshape(p*n, r)
            
            for t in range(r):
                i = t % n
                j = (t // n) % m
                k = (t // (n*m)) % p
                U_part[i*m+j, t] = 1
                V_part[j*p+k, t] = 1
                W_part[k*n+i, t] = 1
            
            x0 += np.random.randn(len(x0)) * 0.2
        else:
            x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try local search on half-integers
                U_half = _snap_to_half(U)
                V_half = _snap_to_half(V)
                W_half = _snap_to_half(W)
                
                U_ls, V_ls, W_ls, ls_err = _local_search_half_integer(
                    U_half, V_half, W_half, n, m, p, 
                    max_iterations=300, 
                    time_limit=min(10, max(1, time_limit - (time.time() - start_time))))
                
                if _verify_decomposition(U_ls, V_ls, W_ls, n, m, p, tol=1e-6):
                    return U_ls, V_ls, W_ls, True
                    
        except Exception:
            pass
            
    if best_x is not None and best_loss < 1e-4:
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_half = _snap_to_half(U)
        V_half = _snap_to_half(V)
        W_half = _snap_to_half(W)
        
        U_ls, V_ls, W_ls, ls_err = _local_search_half_integer(
            U_half, V_half, W_half, n, m, p,
            max_iterations=500,
            time_limit=min(15, max(1, time_limit - (time.time() - start_time))))
        
        if _verify_decomposition(U_ls, V_ls, W_ls, n, m, p, tol=1e-6):
            return U_ls, V_ls, W_ls, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Strategy: Decompose 3x4x6 into two 3x4x3 blocks.
    # If we find a rank r < 36 for 3x4x3, we can form a rank 2r < 72 solution for 3x4x6.
    
    # Try to find low rank for 3x4x3
    # Start from reasonable guesses. Schoolbook is 36.
    # Try ranks 30 to 35.
    for r in range(30, 36):
        if time.time() > deadline - 15:
            break
        
        U_343, V_343, W_343, success = _optimize_rank_343(r, time_limit=15)
        
        if success:
            # Construct 3x4x6 solution
            # U_new: (12, 2r)
            # V_new: (24, 2r)
            # W_new: (18, 2r)
            
            U_new = np.zeros((n*m, 2*r))
            V_new = np.zeros((m*p, 2*r))
            W_new = np.zeros((p*n, 2*r))
            
            # Block 1: k=0,1,2
            U_new[:, 0:r] = U_343
            V_new[0:12, 0:r] = V_343 # m*p for 3x4x3 is 4*3=12
            W_new[0:9, 0:r] = W_343   # p*n for 3x4x3 is 3*3=9
            
            # Block 2: k=3,4,5
            U_new[:, r:2*r] = U_343
            
            for j in range(m):
                for k_prime in range(3):
                    V_new[j*6 + 3 + k_prime, r:2*r] = V_343[j*3 + k_prime, :]
                    
            for k_prime in range(3):
                for i in range(n):
                    W_new[(3+k_prime)*3 + i, r:2*r] = W_343[k_prime*3 + i, :]
            
            # Verify the full 3x4x6 decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U_new]
                V_list = [[float(x) for x in row] for row in V_new]
                W_list = [[float(x) for x in row] for row in W_new]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
