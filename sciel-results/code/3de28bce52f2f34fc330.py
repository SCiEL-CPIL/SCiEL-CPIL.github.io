"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import itertools
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Randomized search for lower rank
    random.seed(42)
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few candidate ranks
    candidate_ranks = list(range(lower_bound, best_rank, 5))
    candidate_ranks.append(best_rank - 1)
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n * m):
                for c in range(target_rank):
                    U[i][c] = random.randint(-1, 1)
            for i in range(m * p):
                for c in range(target_rank):
                    V[i][c] = random.randint(-1, 1)
            for i in range(p * n):
                for c in range(target_rank):
                    W[i][c] = random.randint(-1, 1)
            
            if _check_exact(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
