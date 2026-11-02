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


def _try_merge_columns(U, V, W, n, m, p, deadline):
    """
    Try to merge pairs of columns in the decomposition to reduce rank.
    Two columns c1 and c2 can be merged if we can find a scaling factor
    such that the combined column still satisfies the tensor identity.
    
    This is a heuristic approach: for each pair of columns, check if
    their contributions are 'compatible' in a way that allows merging.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # Try to find linear dependencies or mergeable pairs
    # A simpler approach: check if any column is a scalar multiple of another
    # in a way that allows cancellation
    
    # For now, this is a placeholder for more sophisticated merging
    # In practice, we'd need to solve a system of equations
    pass


def _structured_block_decomposition(n, m, p, deadline):
    """
    Try to decompose the 3x5x6 multiplication into smaller blocks.
    For example, split the 5x6 matrix into 3x6 and 2x6 parts,
    or use Strassen-like tricks on sub-blocks.
    
    This is experimental and may not yield improvements.
    """
    # Placeholder for block-structured search
    pass


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Candidate ranks to try
    candidate_ranks = sorted(set(range(lower_bound, best_rank, 5)) | {lower_bound, lower_bound+1, lower_bound+2, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85})
    
    random.seed(42)
    
    # Try random search with structured patterns
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Each column: pick random support sizes and values
            for c in range(target_rank):
                su = random.randint(1, 4)
                sv = random.randint(1, 4)
                sw = random.randint(1, 4)
                
                for idx in random.sample(range(n * m), min(su, n * m)):
                    U[idx][c] = random.choice([-1, 1])
                for idx in random.sample(range(m * p), min(sv, m * p)):
                    V[idx][c] = random.choice([-1, 1])
                for idx in random.sample(range(p * n), min(sw, p * n)):
                    W[idx][c] = random.choice([-1, 1])
            
            if _check_exact(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
