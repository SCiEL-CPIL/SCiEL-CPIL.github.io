"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 24)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, n, m, p):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W, n, m, p):
        """Check if the numpy decomposition is valid."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0

    def try_als(r, max_restarts=20, max_iters=300):
        """Try to find a valid rank-r decomposition using ALS."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # ALS iterations
            converged = False
            for iteration in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-6:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Round to nearest half-integers and verify
            U_rounded = np.array([[round_to_half(U[i, c]) for c in range(r)] for i in range(n*m)])
            V_rounded = np.array([[round_to_half(V[i, c]) for c in range(r)] for i in range(m*p)])
            W_rounded = np.array([[round_to_half(W[i, c]) for c in range(r)] for i in range(p*n)])
            
            if verify_np(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # Also try without rounding if it's already very close
            if max_err < 1e-8:
                U_list = [[float(U[i, c]) for c in range(r)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(r)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(r)] for i in range(p*n)]
                if verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 11 upward
    for r in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        result = try_als(r)
        if result is not None:
            best_result = result
            best_rank = r
            return best_result
    
    # If all ALS attempts fail, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
