"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    # Known lower bound: max(n*m, n*p, m*p) = max(10, 12, 30) = 30.
    # Estimated rank: m*p + m + p - 1 = 30 + 5 + 6 - 1 = 40.
    # We return the schoolbook decomposition of rank 60 as a safe fallback.
    # Implementing a correct rank-40 decomposition requires a specific construction.
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
