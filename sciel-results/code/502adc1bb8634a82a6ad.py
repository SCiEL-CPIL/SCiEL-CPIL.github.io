import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START

def _schoolbook():
    """Schoolbook decomposition: rank = n*m*p = 50."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n=2, m=5, p=5, trials=10, tol=1e-9):
    """Verify that U,V,W correctly decompose the matmul tensor."""
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_dot = 0.0
                    for j in range(m):
                        u_dot += U[i * m + j, c] * A[i, j]
                    v_dot = 0.0
                    for j in range(m):
                        v_dot += V[j * p + k, c] * B[j, k]
                    val += W[k * n + i, c] * u_dot * v_dot
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _try_rank_33(seed=42, time_limit=10):
    """Attempt to find a rank-33 decomposition."""
    np.random.seed(seed)
    start_time = time.time()
    rank = 33
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
                
    # Simple ALS refinement
    for it in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        A_mat[r, k] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_target[i, j, :], rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        A_mat[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_target[:, j, k], rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        A_mat[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_mat.T, T_target[i, :, k], rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Snap to integers/half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Check error
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        
        error = np.linalg.norm(T_recon - T_target)
        if error < 1e-6:
            return U, V, W
            
    return None

def construct():
    deadline = time.time() + 110 # Safety margin
    
    # Fallback
    U, V, W = _schoolbook()
    best_result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    # Try to find rank 33
    for seed in range(5):
        if time.time() > deadline:
            break
        result = _try_rank_33(seed=seed, time_limit=15)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, trials=5):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
                
    return best_result

# EVOLVE-BLOCK-END
