"""Tensor decomposition search for 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _elapsed():
    return time.time() - START_TIME

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes matrix multiplication."""
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The multiplication tensor T[i,j,k] = 1 if and only if the term contributes
    # For each rank component c, the contribution is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Sum over c should give 1 for all (i,j,k)
    
    # Build the tensor T
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Contribution to T[i,j,k] from component c:
        # u[i,j] * v[j,k] * w[k,i]
        # This is a bit tricky. Let's think.
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # = sum_c u[i,j] * v[j,k] * w[k,i]
        
        for i in range(n):
            for j in range(m):
                if u[i, j] != 0:
                    for k in range(p):
                        if v[j, k] != 0 and w[k, i] != 0:
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-10)

def _try_merge(U, V, W, n, m, p, max_attempts=50):
    """Try to find a lower rank decomposition by merging terms."""
    rank = len(U[0]) if U else 0
    
    # Try to find pairs of terms that can be merged
    # A merge is possible if we can find a new term that replaces two terms
    # This is complex, so let's try a simpler approach:
    # Check if any term is redundant (can be zeroed out)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for attempt in range(max_attempts):
        if _elapsed() > TIME_LIMIT - 5:
            break
            
        # Try removing each term
        for c in range(best_rank):
            if _elapsed() > TIME_LIMIT - 5:
                break
                
            # Try setting term c to zero and see if we can compensate
            # This is a heuristic: check if the sum of all other terms
            # already gives 1 for all (i,j,k) that term c contributes to
            
            # Find which (i,j,k) term c contributes to
            contributed = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = best_U[i*m+j][c] * best_V[j*p+k][c] * best_W[k*n+i][c]
                        if val != 0:
                            contributed.add((i,j,k))
            
            if not contributed:
                # Term c is already zero everywhere, remove it
                new_rank = best_rank - 1
                new_U = [best_U[r][:c] + best_U[r][c+1:] for r in range(len(best_U))]
                new_V = [best_V[r][:c] + best_V[r][c+1:] for r in range(len(best_V))]
                new_W = [best_W[r][:c] + best_W[r][c+1:] for r in range(len(best_W))]
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = new_rank
        
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    result = _schoolbook(n, m, p)
    U, V, W = result["U"], result["V"], result["W"]
    
    # Try to merge terms
    if _elapsed() < TIME_LIMIT - 10:
        U, V, W, rank = _try_merge(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
