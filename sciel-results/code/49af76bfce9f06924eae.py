import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 12.0

    n, m, p = DIMENSIONS
    
    # Strassen 2x2 coefficients
    a_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    b_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    c_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]

    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            for S in range(2):
                for T in range(2):
                    outer_a = a_coeffs[s][S][T]
                    if outer_a == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            inner_a = a_coeffs[t][i2][j2]
                            if inner_a == 0: continue
                            i = 2 * S + i2
                            j = 2 * T + j2
                            U[i * 4 + j, c_idx] += outer_a * inner_a
            for T in range(2):
                for K in range(2):
                    outer_b = b_coeffs[s][T][K]
                    if outer_b == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            inner_b = b_coeffs[t][j2][k2]
                            if inner_b == 0: continue
                            j = 2 * T + j2
                            k = 2 * K + k2
                            V[j * 4 + k, c_idx] += outer_b * inner_b
            for K in range(2):
                for S in range(2):
                    outer_c = c_coeffs[s][S][K]
                    if outer_c == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            inner_c = c_coeffs[t][i2][k2]
                            if inner_c == 0: continue
                            k = 2 * K + k2
                            i = 2 * S + i2
                            W[k * 4 + i, c_idx] += outer_c * inner_c

    # Check for linear dependencies to reduce rank
    # This is a heuristic: check if the last term is dependent on others
    # If we can't prove it quickly, we return 49.
    
    # For now, return 49.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
