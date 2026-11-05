"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n, m, p = 6, 6, 6
    rank = 189

    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)

    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U offsets: 0=(0,0)=A00, 1=(0,1)=A01, 2=(1,0)=A10, 3=(1,1)=A11
    # V offsets: 0=(0,0)=B00, 1=(0,1)=B01, 2=(1,0)=B10, 3=(1,1)=B11
    # W offsets: 0=(0,0)=C00, 1=(0,1)=C01, 2=(1,0)=C10, 3=(1,1)=C11

    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        # M2 = (A10+A11)*B00
        (([2, 3], [1, 1]), ([0], [1])),
        # M3 = A00*(B01-B11)
        (([0], [1]), ([1, 3], [1, -1])),
        # M4 = A11*(B10-B00)
        (([3], [1]), ([2, 0], [1, -1])),
        # M5 = (A00+A01)*B11
        (([0, 1], [1, 1]), ([3], [1])),
        # M6 = (A10-A00)*(B00+B01)
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        # M7 = (A01-A11)*(B10+B11)
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]

    # W contributions for each Strassen term:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contributions = [
        [(0, 1), (3, 1)],   # M1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],  # M2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],   # M3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],   # M4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],  # M5 -> C00(-1), C01(+1)
        [(3, 1)],           # M6 -> C11(+1)
        [(0, 1)],           # M7 -> C00(+1)
    ]

    col = 0
    for bi in range(3):  # block row of A (0,1,2)
        for bk in range(3):  # block col of A = block row of B (0,1,2)
            for bj in range(3):  # block col of B (0,1,2)
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]

                    # Set U factors: A element at local (r,c) in block (bi,bk)
                    # global row = 2*bi + r, global col = 2*bk + c
                    # U index = global_row * 6 + global_col
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff

                    # Set V factors: B element at local (r,c) in block (bk,bj)
                    # global row = 2*bk + r, global col = 2*bj + c
                    # V index = global_row * 6 + global_col
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff

                    # Set W factors: C element at local (r,c) in block (bi,bj)
                    # global row = 2*bi + r, global col = 2*bj + c
                    # W index = global_col * 6 + global_row  (W[k*n+i] convention)
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bj + c
                        W[global_col * 6 + global_row][col] += coeff

                    col += 1

    if time.time() > deadline:
        return {"U": U, "V": W, "W": W}  # fallback, should not happen

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
