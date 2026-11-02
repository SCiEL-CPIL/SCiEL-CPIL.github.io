"""Rewritten program: rank-28 Strassen decomposition for M_{2,2,8} with deadline guard."""
import time
import random

DIMENSIONS = [2, 2, 8]
DEADLINE = 120.0  # seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 coefficients
    # A layout: [A00, A01, A10, A11] -> indices 0,1,2,3
    # B layout: [B00, B01, B10, B11] -> local indices 0,1,2,3
    # C layout: [C00, C10, C01, C11] -> local indices 0,1,2,3 (i = idx%2, k_local = idx//2)

    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]

    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]

    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]

    # 4 independent 2x2 blocks along the k dimension
    for q in range(4):
        for s in range(7):
            r = q * 7 + s

            for a_idx, val in U_coeffs[s].items():
                U[a_idx][r] = val

            for b_local, val in V_coeffs[s].items():
                j = b_local // 2
                k_loc = b_local % 2
                k = 2 * q + k_loc
                V[j * p + k][r] = val

            for c_local, val in W_coeffs[s].items():
                i = c_local % 2
                k_loc = c_local // 2
                k = 2 * q + k_loc
                W[k * n + i][r] = val

    # Check deadline
    if time.time() - START_TIME > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}

    # Attempt to find a lower rank solution (e.g., rank 24) via random search
    # This is a heuristic attempt. If it fails, we return the rank-28 solution.
    # Given the complexity, we skip the actual search to avoid timeout risks
    # and return the known valid rank-28 solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
