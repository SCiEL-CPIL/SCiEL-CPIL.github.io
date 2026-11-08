"""Matrix multiplication 2,3,3 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 18
    
    def build_schoolbook():
        rank = rank_schoolbook
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
    
    U, V, W = build_schoolbook()
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank using alternating least squares with multiple random restarts
    def try_rank_als(r, n_restarts=20, max_iter=3000, seed_base=42):
        """Try to find rank-r decomposition using ALS."""
        if time.time() > deadline - 15.0:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                return None
            
            rng = np.random.RandomState(seed_base + restart)
            U_mat = rng.randn(n * m, r) * 0.5
            V_mat = rng.randn(m * p, r) * 0.5
            W_mat = rng.randn(p * n, r) * 0.5
            
            for iteration in range(max_iter):
                if time.time() > deadline - 2.0:
                    return None
                
                # Update U: for each (i, j), solve A @ U_row = ones(p)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            for rr in range(r):
                                A[k, rr] = V_mat[j * p + k, rr] * W_mat[k * n + i, rr]
                        try:
                            U_mat[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j, k), solve B @ V_row = ones(n)
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, r))
                        for i in range(n):
                            for rr in range(r):
                                B[i, rr] = U_mat[i * m + j, rr] * W_mat[k * n + i, rr]
                        try:
                            V_mat[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k, i), solve C @ W_row = ones(m)
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, r))
                        for j in range(m):
                            for rr in range(r):
                                C[j, rr] = U_mat[i * m + j, rr] * V_mat[j * p + k, rr]
                        try:
                            W_mat[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if iteration % 50 == 0:
                    if verify(U_mat, V_mat, W_mat, tol=1e-8):
                        U_rounded = np.round(U_mat * 2) / 2
                        V_rounded = np.round(V_mat * 2) / 2
                        W_rounded = np.round(W_mat * 2) / 2
                        if verify(U_rounded, V_rounded, W_rounded, tol=1e-9):
                            return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 9 to 17
    for r in range(9, 18):
        if time.time() > deadline - 30.0:
            break
        result = try_rank_als(r, n_restarts=10, max_iter=2000, seed_base=r * 100)
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try, tol=1e-9):
                best_rank = r
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
