"""Matrix multiplication 2x3x6 tensor decomposition; structured rank-12 attempt."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s (150s wall)
    n, m, p = DIMENSIONS

    # Strategy: decompose 2x3x6 as direct sum of m=3 sub-problems 2x1x6.
    # Each 2x1x6 has schoolbook rank 12. Total = 3*12 = 36 (no gain).
    # 
    # Alternative: use the known rank-4 decomposition of 2x1x2 and tensor
    # with a rank-3 structure for the width-3 direction.
    #
    # The 2x1x2 matrix multiplication tensor T has rank 4 (schoolbook is optimal
    # for small sizes). Its 4 rank-1 terms can be written as:
    #   T = sum_{r=1}^{4} u_r ⊗ v_r ⊗ w_r
    # where for 2x1x2:
    #   U (2x1) entries, V (1x2) entries, W (2x2) entries
    #
    # A known rank-4 decomposition of 2x1x2 (which is just schoolbook):
    #   C[0,0] = A[0,0]*B[0,0]
    #   C[0,1] = A[0,0]*B[0,1]
    #   C[1,0] = A[1,0]*B[0,0]
    #   C[1,1] = A[1,0]*B[0,1]
    #
    # For 2x1x6, we can view B as 3 copies of 1x2 (or use other groupings).
    # But 2x1x6 schoolbook = 12, and lower bound = max(2*6, 1*6, 2*1*6/...) 
    # Actually lower bound for 2x1x6: output has 12 entries, so rank >= 12.
    # So 2x1x6 is rank 12 exactly.
    #
    # Total for 2x3x6 via direct sum: 3 * 12 = 36. No improvement.
    #
    # However, let's try a different structural approach: exploit that
    # the middle dimension m=3 allows sharing across j indices.
    #
    # For now, return the verified schoolbook decomposition of rank 36.
    # Future: search for rank < 36 using structured ansatz.

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
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
