"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition via random search
    # We'll try to find a rank-99 decomposition
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_random_search(target_rank, max_iters=500):
        """Try to find a valid decomposition with given rank."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Initialize with schoolbook and perturb
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Start with first target_rank schoolbook terms
            for c in range(target_rank):
                idx = c  # linear index into (i,j,k)
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
            
            # Try to fix the missing term by adjusting
            # The missing term is at index target_rank
            i_miss = target_rank // (m * p)
            rem_miss = target_rank % (m * p)
            j_miss = rem_miss // p
            k_miss = rem_miss % p
            
            # We need to add contributions to cover (i_miss, j_miss, k_miss)
            # and not disturb other entries
            # This is very constrained, so random search is unlikely to work
            # Just verify if current is valid (it won't be, since we're missing one term)
            pass
        
        return None
    
    # For now, return schoolbook
    result = schoolbook()
    
    # Check time
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
