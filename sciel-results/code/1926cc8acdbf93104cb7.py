import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Schoolbook rank is n*m*p = 45
    # We want to find a lower rank decomposition
    
    # Strategy: Use the known rank-7 algorithm for 3x3x3 (Strassen-like)
    # and combine with schoolbook for the remaining 2 columns
    # Total rank: 7 + 3*3*2 = 7 + 18 = 25
    
    # But we can do better by finding a direct decomposition
    
    # Let's try to find a decomposition of rank R where R < 45
    # We'll use a randomized approach with local optimization
    
    def _compute_tensor(U, V, W, n, m, p):
        """Compute tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    if u[i, j] != 0:
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return T
    
    def _compute_tensor_fast(U, V, W, n, m, p):
        """Vectorized tensor computation"""
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return T
    
    def _verify(U, V, W, n, m, p, tol=1e-8):
        T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
        T_target = np.ones((n, m, p))
        return np.allclose(T_decomp, T_target, atol=tol)
    
    # Try to find a low-rank decomposition
    # Start with a reasonable guess and use optimization
    
    best_U, best_V, best_W, best_rank = None, None, None, n * m * p  # Start with schoolbook
    
    # First, let's try a structured approach
    # Use the fact that we can decompose 3x3x5 as sum of 3x3x3 + 3x3x2
    # For 3x3x3, use rank-7 (if we can find it)
    # For 3x3x2, use schoolbook rank 18
    # Total: 7 + 18 = 25
    
    # Let's try to construct a rank-25 decomposition directly
    target_rank = 25
    
    # Initialize with random values
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Initialize with small random values
        for c in range(target_rank):
            # Make each term sparse
            n_nz_U = np.random.randint(2, 5)
            n_nz_V = np.random.randint(2, 5)
            n_nz_W = np.random.randint(2, 5)
            
            idx_U = np.random.choice(n * m, size=n_nz_U, replace=False)
            idx_V = np.random.choice(m * p, size=n_nz_V, replace=False)
            idx_W = np.random.choice(p * n, size=n_nz_W, replace=False)
            
            for idx in idx_U:
                U[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in idx_V:
                V[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in idx_W:
                W[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        # Try to optimize
        T_target = np.ones((n, m, p))
        best_error = np.inf
        best_U_copy, best_V_copy, best_W_copy = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
            error = np.max(np.abs(T_decomp - T_target))
            
            if error < 1e-8:
                best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), target_rank
                break
            
            if error < best_error:
                best_error = error
                best_U_copy, best_V_copy, best_W_copy = U.copy(), V.copy(), W.copy()
            
            improved = False
            for c in range(target_rank):
                for delta in [-0.5, 0.5]:
                    # Try U
                    U_old = U[:, c].copy()
                    U[:, c] += delta
                    T_new = _compute_tensor_fast(U, V, W, n, m, p)
                    err_new = np.max(np.abs(T_new - T_target))
                    if err_new < error:
                        error = err_new
                        improved = True
                    else:
                        U[:, c] = U_old
                    
                    # Try V
                    V_old = V[:, c].copy()
                    V[:, c] += delta
                    T_new = _compute_tensor_fast(U, V, W, n, m, p)
                    err_new = np.max(np.abs(T_new - T_target))
                    if err_new < error:
                        error = err_new
                        improved = True
                    else:
                        V[:, c] = V_old
                    
                    # Try W
                    W_old = W[:, c].copy()
                    W[:, c] += delta
                    T_new = _compute_tensor_fast(U, V, W, n, m, p)
                    err_new = np.max(np.abs(T_new - T_target))
                    if err_new < error:
                        error = err_new
                        improved = True
                    else:
                        W[:, c] = W_old
            
            if not improved:
                break
        
        if best_U is not None:
            break
    
    # If we didn't find a good solution, fall back to schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n * m, rank), dtype=np.float64)
        best_V = np.zeros((m * p, rank), dtype=np.float64)
        best_W = np.zeros((p * n, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, c] = 1.0
                    best_V[j * p + k, c] = 1.0
                    best_W[k * n + i, c] = 1.0
                    c += 1
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
