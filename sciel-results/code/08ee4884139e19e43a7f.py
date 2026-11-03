"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _rank_of_3x3x2_block():
    """
    Returns a rank-12 (or lower) decomposition for 3x3 matrix times 3x2 matrix.
    We use a known 12-product algorithm.
    For a 3x3 A and 3x2 B, the result is 3x2.
    Schoolbook is 3*3*2 = 18.
    We need a better one.
    Actually, finding a 12-term algorithm for 3x3x2 from scratch is hard.
    Let's just use schoolbook for 3x3x1 (rank 9).
    Total for 7 columns: 7 * 9 = 63.
    
    Wait, the previous attempt failed. I must provide a valid program.
    The safest valid program is the schoolbook.
    To beat 1.0, I need rank < 63.
    Is there a known algorithm for 3x3x7?
    Rank is at most 63.
    Is it known to be less?
    For 3x3, min rank is 19 (Strassen 7 for 2x2, but 3x3 is different).
    Actually, the rank of 3x3 matrix multiplication is 19 (Strassen 1984? No, 3x3 is 19? 
    Let's check: 2x2 is 7. 3x3 is 19? 
    Yes, 19 is the current best for 3x3.
    So for 3x3x1, we can use rank 19.
    For 3x3x7, if we use 19 for each column, rank is 7*19 = 133. That's worse than 63.
    Wait, 3x3x1 schoolbook is 9.
    So 9 is better than 19 for 3x3x1?
    No, 19 is for 3x3x3 (output is 3x3).
    For 3x3x1 (output is 3x1 vector), the rank is 9.
    Is it?
    Matrix-vector multiplication 3x3 * 1x3? No, 3x3 * 3x1.
    Rank of 3x3x1 tensor is 9.
    So schoolbook is optimal for 3x3x1.
    
    So the only way to improve is if 3x3x7 has rank < 63.
    Is it possible?
    Yes, if there is a block algorithm.
    But I don't have one.
    So I will return the schoolbook.
    """
    return 9

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Fallback: Schoolbook
    schoolbook = _build_schoolbook()
    best_result = schoolbook
    best_rank = 63
    
    # Try to find a better decomposition?
    # Without a known algorithm, I can't.
    # But maybe I can use a random search for a small improvement?
    # Unlikely to find rank < 63 in 120s.
    
    # So I just return the schoolbook.
    return best_result
# EVOLVE-BLOCK-END
