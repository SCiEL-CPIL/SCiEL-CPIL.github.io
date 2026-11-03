"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=3000, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.3
        V = np.random.randn(m*p, r) * 0.3
        W = np.random.randn(p*n, r) * 0.3
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            if verify(U, V, W):
                return U, V, W, r, True
        
        return U, V, W, r, False
    
    # Try to find lower rank
    for r in [16, 17]:
        found = False
        for seed in range(30):
            U_try, V_try, W_try, rank_try, success = try_rank(r, max_iters=2000, seed=seed)
            if success:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, rank_try
                found = True
                break
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
