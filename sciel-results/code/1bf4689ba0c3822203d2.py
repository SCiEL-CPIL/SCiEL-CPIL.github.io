"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
TIME_LIMIT = 150
SAFETY_MARGIN = 5

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Trivial rank
    rank_trivial = n * m * p
    
    # For 2x2 * 2x14, let's try a rank-28 decomposition.
    # 
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as a sum of rank-1 terms.
    # 
    # For each k, we have a 2x2 * 2x1 subproblem.
    # The rank of 2x2 * 2x1 is 4.
    # So for 14 values of k, the total rank is 14*4 = 56.
    # 
    # To get rank 28, we need to share multiplications across different k values.
    # 
    # Let's try a random search.
    
    def try_rank(r, max_restarts=10, max_iters=200):
        best_error = float('inf')
        best_result = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iters):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                for a in range(p * n):
                    i = a % n
                    k = a // n
                    A_eq = np.zeros((m, r))
                    b_eq = np.zeros(m)
                    for j in range(m):
                        b = i * m + j
                        c = j * p + k
                        A_eq[j, :] = U[b, :] * V[c, :]
                        b_eq[j] = 1.0
                    W[a, :] = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
                
                for b in range(n * m):
                    i = b // m
                    j = b % m
                    A_eq = np.zeros((p, r))
                    b_eq = np.zeros(p)
                    for k in range(p):
                        a = k * n + i
                        c = j * p + k
                        A_eq[k, :] = W[a, :] * V[c, :]
                        b_eq[k] = 1.0
                    U[b, :] = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
                
                for c in range(m * p):
                    j = c // p
                    k = c % p
                    A_eq = np.zeros((n, r))
                    b_eq = np.zeros(n)
                    for i in range(n):
                        a = k * n + i
                        b = i * m + j
                        A_eq[i, :] = W[a, :] * U[b, :]
                        b_eq[i] = 1.0
                    V[c, :] = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = k * n + i
                        b = i * m + j
                        c = j * p + k
                        val = np.sum(W[a, :] * U[b, :] * V[c, :])
                        error += abs(val - 1.0)
            
            if error < best_error:
                best_error = error
                best_result = (U.copy(), V.copy(), W.copy())
        
        if best_error < 1e-6 and best_result is not None:
            return best_result
        
        return None
    
    # Try ranks from 28 up to the trivial rank
    for r in range(28, rank_trivial + 1):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        result = try_rank(r, max_restarts=5, max_iters=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to trivial decomposition
    rank = rank_trivial
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
