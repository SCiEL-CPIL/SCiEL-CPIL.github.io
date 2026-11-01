import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)  # 16 x 49
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 49
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 49

    # Define the 7 Strassen terms for a 2x2 product
    # Each term is (A_mask, B_mask, C_mask)
    # Masks are dictionaries mapping (row, col) to coefficient
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 -= M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # M2 = (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00 * (B01 - B11) -> C01 += M3, C11 += M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11 * (B10 - B00) -> C00 += M4, C10 += M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10 - A00) * (B00 + B01) -> C11 += M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t in range(7):
        block_A_t, block_B_t, block_C_t = strassen_terms[t]
        for s in range(7):
            block_A_s, block_B_s, block_C_s = strassen_terms[s]
            r_idx = t * 7 + s

            # U construction
            for (bi, bj), coeff_t in block_A_t.items():
                for (si, sj), coeff_s in block_A_s.items():
                    a_row = 2 * bi + si
                    a_col = 2 * bj + sj
                    U[a_row * 4 + a_col, r_idx] += coeff_t * coeff_s

            # V construction
            for (bi, bj), coeff_t in block_B_t.items():
                for (si, sj), coeff_s in block_B_s.items():
                    b_row = 2 * bi + si
                    b_col = 2 * bj + sj
                    V[b_row * 4 + b_col, r_idx] += coeff_t * coeff_s

            # W construction
            # Convention: W[k*n+i, r] where k is col of C, i is row of C?
            # Prompt: "W[k*n+i,r]" and "C[i,j] = sum_k A[i,k] B[k,j]"
            # Usually W is the output factor such that C = W * (U^T A) * (B^T V)?
            # Or C = sum_r W_r * (U_r^T A) * (B^T V_r)?
            # The prompt says: "W[k*n+i,r]".
            # In the current code: w_idx = c_col * 4 + c_row.
            # c_col is j, c_row is i.
            # So W[j*4 + i, r].
            # This matches "W[k*n+i,r]" if k=j and n=4.
            for (ci, cj), coeff_t in block_C_t.items():
                for (si, sj), coeff_s in block_C_s.items():
                    c_row = 2 * ci + si
                    c_col = 2 * cj + sj
                    w_idx = c_col * 4 + c_row
                    W[w_idx, r_idx] += coeff_t * coeff_s

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
