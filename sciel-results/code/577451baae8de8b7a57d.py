import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    Rank = 42.
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _try_rank_36_gd():
    """
    Attempt to find a rank 36 decomposition via Gradient Descent (L-BFGS-B).
    """
    n, m, p = 2, 4, 6
    rank = 36
    
    # Precompute target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    target_flat = target.reshape(-1)
    
    def objective_and_grad(x):
        # x is flattened U, V, W
        U_flat, V_flat, W_flat = x[:n*m*rank], x[n*m*rank:n*m*rank+m*p*rank], x[n*m*rank+m*p*rank:]
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        diff = recon - target
        error = np.sum(diff**2)
        
        # Gradient
        # dE/dU[a,c] = 2 * sum_{b,d} diff[a,b,d] V[b,c] W[d,c]
        grad_U = 2 * np.einsum('abd,bc,dc->ac', diff, V, W)
        grad_V = 2 * np.einsum('abd,ac,dc->bc', diff, U, W)
        grad_W = 2 * np.einsum('abd,ac,bc->dc', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad

    # Initialize from Rank 42 solution truncated? Or random?
    # Random is safer to avoid local minima of the truncated solution.
    # But Rank 42 is close. Let's try random first, maybe 2 seeds.
    
    best_result = None
    best_error = float('inf')
    
    for seed in range(2):
        if _deadline_reached():
            break
            
        rng = np.random.RandomState(seed)
        
        # Initialize with small noise
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'maxfun': 200})
            
            if res.success or res.fun < best_error:
                best_error = res.fun
                if best_error < 1e-9:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    return U_opt, V_opt, W_opt
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_decomposition()
    
    # Convert to lists for return
    U42_list = U42.tolist()
    V42_list = V42.tolist()
    W42_list = W42.tolist()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48 if Strassen fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 3. Try to find Rank 36 if time permits
    if _elapsed() < TIME_LIMIT - 20:
        res_36 = _try_rank_36_gd()
        if res_36 is not None:
            U36, V36, W36 = res_36
            is_valid_36, _ = _verify_decomposition(U36, V36, W36)
            if is_valid_36:
                return {
                    "U": U36.tolist(),
                    "V": V36.tolist(),
                    "W": W36.tolist()
                }
    
    # 4. Return Rank 42
    return {"U": U42_list, "V": V42_list, "W": W42_list}
