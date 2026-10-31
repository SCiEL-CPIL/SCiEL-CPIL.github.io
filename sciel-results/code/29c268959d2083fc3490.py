import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = DIMENSIONS
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen coefficients for 2x2 block decomposition
    # a_coeffs[s] is a 2x2 matrix of coefficients for A
    # b_coeffs[s] is a 2x2 matrix of coefficients for B
    # c_coeffs[s] is a 2x2 matrix of coefficients for C
    
    # Term 0: M1 = (A00+A11)(B00+B11) -> C00+C11
    # Term 1: M2 = (A10+A11)B00 -> C10-C11
    # Term 2: M3 = A00(B01-B11) -> C01+C11
    # Term 3: M4 = A11(B10-B00) -> C00+C10
    # Term 4: M5 = (A00+A01)B11 -> -C00+C01
    # Term 5: M6 = (A10-A00)(B00+B10) -> C11
    # Term 6: M7 = (A01-A11)(B01+B11) -> C00

    a_coeffs = [
        [[1, 0], [0, 1]],   # A00+A11
        [[0, 0], [1, 1]],   # A10+A11
        [[1, 0], [0, 0]],   # A00
        [[0, 0], [0, 1]],   # A11
        [[1, 1], [0, 0]],   # A00+A01
        [[-1, 0], [1, 0]],  # A10-A00
        [[0, 1], [0, -1]],  # A01-A11
    ]

    b_coeffs = [
        [[1, 0], [0, 1]],   # B00+B11
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # B01-B11
        [[-1, 0], [1, 0]],  # B10-B00
        [[0, 0], [0, 1]],   # B11
        [[1, 0], [1, 0]],   # B00+B10 (Corrected)
        [[0, 1], [0, 1]],   # B01+B11 (Corrected)
    ]

    c_coeffs = [
        [[1, 0], [0, 1]],   # C00+C11
        [[0, 0], [1, -1]],  # C10-C11
        [[0, 1], [0, 1]],   # C01+C11
        [[1, 0], [1, 0]],   # C00+C10
        [[-1, 1], [0, 0]],  # -C00+C01
        [[0, 0], [0, 1]],   # C11
        [[1, 0], [0, 0]],   # C00
    ]

    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # U: A coefficients
            # U[i, j, c_idx] = a_coeffs[s][i//2][j//2] * a_coeffs[t][i%2][j%2]
            for i in range(4):
                for j in range(4):
                    val = a_coeffs[s][i // 2][j // 2] * a_coeffs[t][i % 2][j % 2]
                    if val != 0:
                        U[i * 4 + j, c_idx] = val

            # V: B coefficients
            # V[j, k, c_idx] = b_coeffs[s][j//2][k//2] * b_coeffs[t][j%2][k%2]
            for j in range(4):
                for k in range(4):
                    val = b_coeffs[s][j // 2][k // 2] * b_coeffs[t][j % 2][k % 2]
                    if val != 0:
                        V[j * 4 + k, c_idx] = val

            # W: C coefficients
            # W[k, i, c_idx] = c_coeffs[s][k//2][i//2] * c_coeffs[t][k%2][i%2]
            # Note: W is indexed by k*n+i, so row is k*4+i
            for k in range(4):
                for i in range(4):
                    val = c_coeffs[s][k // 2][i // 2] * c_coeffs[t][k % 2][i % 2]
                    if val != 0:
                        W[k * 4 + i, c_idx] = val

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
