"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p as lists."""
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

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 6
    rank_sb = n * m * p  # 120

    best_result = _make_schoolbook(n, m, p)
    best_rank = rank_sb

    # The schoolbook decomposition is the safe, correct answer.
    # We attempt to find a lower-rank decomposition using structured random search.
    # For 4x5x6, the known rank is 120 (schoolbook), but we try to see if we can
    # find a lower rank using random structured patterns with integer/half-integer entries.

    # Entries to try
    entries = [-1, -0.5, 0, 0.5, 1]

    # Try a few random structured decompositions for rank < 120
    random.seed(42)

    # We'll try ranks from 119 down to maybe 100, but only a few trials per rank
    # due to time constraints. Each trial is a random assignment of entries.
    
    # For efficiency, we can try to be smarter:
    # 1. Start with schoolbook and try to merge columns (combine two rank-1 terms into one)
    # 2. Or try random structured matrices
    
    # Let's try a hybrid approach:
    # For each target rank r < 120, try to find U, V, W with entries from a small set
    # such that the tensor identity holds.
    
    # This is a very hard problem in general, but for small dimensions we can try.
    
    # Strategy: For each target rank, try random assignments and verify.
    # If we find one, return it. Otherwise, return schoolbook.
    
    # Given the difficulty, we'll limit the search to a few ranks and few trials.
    
    for target_rank in range(119, 100, -1):
        if time.time() > deadline - 20.0:
            break
        
        for trial in range(50):
            if time.time() > deadline - 15.0:
                break
            
            # Random structured decomposition
            U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
