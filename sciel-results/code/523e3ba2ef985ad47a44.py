"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses alternating minimization to find low-rank decompositions.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s < 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_alternating_minimization(rank, max_iters=300, tol=1e-10, seed=0):
        """Try alternating minimization to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check residual every 5 iterations
            if it % 5 == 4 or it == max_iters - 1:
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_res = max(max_res, abs(s - 1.0))
                if max_res < tol:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try alternating minimization for various ranks
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something
    # For 2x4x7: try from 28 upward
    ranks_to_try = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = None
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            found = try_alternating_minimization(r, max_iters=200, seed=seed)
            if found is not None:
                best_rank = r
                best_result = found
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
