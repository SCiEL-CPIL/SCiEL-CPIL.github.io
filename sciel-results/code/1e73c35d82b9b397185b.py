"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 via Strassen on 2x2 blocks."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 140

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for b in range(2):
            for a_blk in range(2):
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1

                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff

                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff

                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff

                    col += 1

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Build the rank-28 Strassen decomposition
    result = build_strassen_28()

    # Validate to ensure correctness
    if not validate(result["U"], result["V"], result["W"]):
        raise ValueError("Strassen-28 decomposition failed validation")

    return result

# EVOLVE-BLOCK-END
