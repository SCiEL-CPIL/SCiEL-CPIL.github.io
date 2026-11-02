"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try to find a better decomposition using block structure
    # Split 5 = 3 + 2
    n1, n2 = 3, 2
    m1, m2 = 3, 2
    p1, p2 = 3, 2
    
    # We'll build U, V, W matrices
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # For each block product, we need to add rank
    # Block products needed:
    # C11 = A11*B11 + A12*B21  (3x3 * 3x3 + 3x2 * 2x3)
    # C12 = A11*B12 + A12*B22  (3x3 * 3x2 + 3x2 * 2x2)
    # C21 = A21*B11 + A22*B21  (2x3 * 3x3 + 2x2 * 2x3)
    # C22 = A21*B12 + A22*B22  (2x3 * 3x2 + 2x2 * 2x2)
    
    # Helper: get rank for multiplying (a x b) by (b x c)
    # Use known values or upper bounds
    def get_rank(a, b, c):
        # Known exact values
        if a == 1 and b == 1 and c == 1:
            return 1
        if a == 2 and b == 2 and c == 2:
            return 7
        if a == 3 and b == 3 and c == 3:
            return 19
        # Upper bound: pad to max dimension and use that rank
        max_dim = max(a, b, c)
        # Conservative upper bound
        if max_dim <= 2:
            return 7
        if max_dim <= 3:
            return 19
        # Fallback: schoolbook
        return a * b * c
    
    # Compute total rank for block decomposition
    rank_33_33 = get_rank(3, 3, 3)  # 19
    rank_32_23 = get_rank(3, 2, 3)  # M(3,2,3)
    rank_33_32 = get_rank(3, 3, 2)  # M(3,3,2)
    rank_32_22 = get_rank(3, 2, 2)  # M(3,2,2)
    rank_23_33 = get_rank(2, 3, 3)  # M(2,3,3)
    rank_22_23 = get_rank(2, 2, 3)  # M(2,2,3)
    rank_23_32 = get_rank(2, 3, 2)  # M(2,3,2)
    rank_22_22 = get_rank(2, 2, 2)  # 7
    
    # For mixed dimensions, we can often do better than padding
    # M(3,2,3): multiply 3x2 by 2x3. This is like computing 9 entries, each is a dot product of length 2.
    # Rank is at most 3*2*3 = 18, but likely less.
    # Actually, M(a,b,c) ≤ M(a,b,c') if c ≤ c', etc.
    # M(3,2,3) ≤ M(3,3,3) = 19, but also ≤ M(3,2,2) * something...
    # Let's be conservative and use the schoolbook for mixed unless we know better.
    
    # For now, use conservative estimates
    rank_32_23 = 3 * 2 * 3  # 18
    rank_33_32 = 3 * 3 * 2  # 18
    rank_32_22 = 3 * 2 * 2  # 12
    rank_23_33 = 2 * 3 * 3  # 18
    rank_22_23 = 2 * 2 * 3  # 12
    rank_23_32 = 2 * 3 * 2  # 12
    
    total_rank = (rank_33_33 + rank_32_23 +  # C11
                  rank_33_32 + rank_32_22 +  # C12
                  rank_23_33 + rank_22_23 +  # C21
                  rank_23_32 + rank_22_22)   # C22
    
    total_rank = (19 + 18 +  # C11
                  18 + 12 +  # C12
                  18 + 12 +  # C21
                  12 + 7)    # C22
    
    total_rank = 19 + 18 + 18 + 12 + 18 + 12 + 12 + 7
    total_rank = 116
    
    # This is better than 125! But I need to actually construct the decomposition.
    # For now, if the computed rank is less than 125, I'll try to construct it.
    # Otherwise, fall back to schoolbook.
    
    if total_rank >= 125:
        # Fall back to schoolbook
        rank = 125
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
    
    # For now, return schoolbook as a safe fallback
    # TODO: Actually construct the block decomposition
    rank = 125
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
# EVOLVE-BLOCK-END
