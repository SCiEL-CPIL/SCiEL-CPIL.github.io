import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Known optimal rank 35 solution using Strassen for each 2x2 block
    rank_fallback = 35
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]

    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U_fallback[idx][col] = a_vec[idx]
            V_fallback[0 * p + 2 * k][col] = b_vec[0]
            V_fallback[0 * p + 2 * k + 1][col] = b_vec[1]
            V_fallback[1 * p + 2 * k][col] = b_vec[2]
            V_fallback[1 * p + 2 * k + 1][col] = b_vec[3]
            W_fallback[2 * k * n + 0][col] = c_vec[0]
            W_fallback[2 * k * n + 1][col] = c_vec[1]
            W_fallback[(2 * k + 1) * n + 0][col] = c_vec[2]
            W_fallback[(2 * k + 1) * n + 1][col] = c_vec[3]
            col += 1

    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
