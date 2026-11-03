import time
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 115.0

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error_fast(U, V, W, T):
    """
    Compute ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||_F^2
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = reconstructed - T
    return np.sum(diff * diff)

def _als_step_fast(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration.
    """
    rank = U.shape[1]
    
    # Pre-compute index mappings for efficiency
    # We solve for U, V, W in sequence.
    
    # Update U
    # T.reshape(n*m, -1) = U @ (V[:,r] * W[:,r]).reshape(-1)
    # A_u[r, idx] = V[j,r] * W[k,r] where idx corresponds to flattened (j, k) for fixed i?
    # No, T is 3D. We flatten the other two modes.
    # T_flat_u has shape (n*m, m*p*p*n)
    # Index (i*m+j, (j*p+k)*(p*n) + (k*n+i))
    # Coefficient for U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r]
    
    A_u = np.zeros((rank, m*p*p*n), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = (j*p + k) * (p*n) + (k*n + i)
                    A_u[r, idx_u] = V[j*p + k, r] * W[k*n + i, r]
                    
    T_flat_u = T.reshape(n*m, m*p*p*n)
    # Solve A_u X = T_flat_u.T
    U_new, _, _, _ = lstsq(A_u, T_flat_u.T, cond=None)
    U = U_new.T
    
    # Update V
    # T.transpose(1,0,2).reshape(m*p, -1)
    # Index (j*p+k, (i*m+j)*(p*n) + (k*n+i))
    # Coefficient for V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
    
    A_v = np.zeros((rank, n*m*p*n), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_v = (i*m + j) * (p*n) + (k*n + i)
                    A_v[r, idx_v] = U[i*m + j, r] * W[k*n + i, r]
                    
    T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    V_new, _, _, _ = lstsq(A_v, T_flat_v.T, cond=None)
    V = V_new.T
    
    # Update W
    # T.transpose(2,0,1).reshape(p*n, -1)
    # Index (k*n+i, (i*m+j)*(m*p) + (j*p+k))
    # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
    
    A_w = np.zeros((rank, n*m*m*p), dtype=np.float64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_w = (i*m + j) * (m*p) + (j*p + k)
                    A_w[r, idx_w] = U[i*m + j, r] * V[j*p + k, r]
                    
    T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    W_new, _, _, _ = lstsq(A_w, T_flat_w.T, cond=None)
    W = W_new.T
    
    return U, V, W

def _polish_solution(U, V, W, T, time_limit=5.0):
    """
    Use scipy.optimize.minimize to polish the solution if it's close.
    """
    start_time = time.time()
    rank = U.shape[1]
    n, m, p = T.shape
    
    # Flatten U, V, W into a single vector
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10
        U_curr = x[:n*m*rank].reshape(n*m, rank)
        V_curr = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_curr = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute error
        reconstructed = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr, optimize=True)
        diff = reconstructed - T
        return np.sum(diff * diff)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-15})
        if res.success or (time.time() - start_time < time_limit):
            x_opt = res.x
            U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
            V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
            err = objective(x_opt)
            if err < 1e-10:
                return U_opt, V_opt, W_opt, err
    except Exception:
        pass
    return None

def _try_rank_als(n, m, p, rank, T, seed=0, time_limit=5.0):
    start_time = time.time()
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    rng = np.random.RandomState(seed)
    
    # Try a few random initializations
    for init_idx in range(5):
        if time.time() - start_time > time_limit:
            break
            
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize
        for mat in [U, V, W]:
            norms = np.linalg.norm(mat, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms
            
        cur_err = _tensor_error_fast(U, V, W, T)
        
        max_iters = 200
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = _als_step_fast(U, V, W, T, n, m, p)
            cur_err = _tensor_error_fast(U, V, W, T)
            
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-12:
                return best_U, best_V, best_W, best_err
            
            # If close, try polishing
            if best_err < 1e-6 and time.time() - start_time < time_limit - 2:
                polished = _polish_solution(U, V, W, T, time_limit=2.0)
                if polished:
                    U_p, V_p, W_p, err_p = polished
                    if err_p < best_err:
                        best_err = err_p
                        best_U, best_V, best_W = U_p, V_p, W_p
                        if best_err < 1e-10:
                            return best_U, best_V, best_W, best_err
    
    if best_U is None:
        return None
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    T = _build_tensor(n, m, p)
    
    # Known minimal rank for 2x5x5 is 41
    target_rank = 41
    
    # First, try to find the rank-41 solution
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try rank 41 with multiple seeds
    for seed in range(10):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        try:
            res = _try_rank_als(n, m, p, target_rank, T, seed=seed, time_limit=5.0)
            if res:
                U_new, V_new, W_new, err = res
                if err < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
        except Exception:
            continue
    
    # If rank 41 failed, fall back to schoolbook (rank 50)
    if best_rank == n * m * p:
        # Construct schoolbook solution
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
