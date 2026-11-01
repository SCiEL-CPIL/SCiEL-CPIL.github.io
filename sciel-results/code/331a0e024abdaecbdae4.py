import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    # Use numpy arrays for cleaner construction
    U = np.zeros((n * m, rank), dtype=int)  # 16 x 49
    V = np.zeros((m * p, rank), dtype=int)  # 16 x 49
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 49

    # Define the 7 Strassen terms for a 2x2 product
    # Each term: (A_coeffs, B_coeffs, C_contrib)
    # Keys are (row, col) indices within the 2x2 block
    strassen_terms = [
        # M1 = (a+d)(e+h) -> contributes to C11, C22
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (c+d)*e -> contributes to C21, C22
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a*(f-h) -> contributes to C12, C22
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = d*(g-e) -> contributes to C11, C21
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a+b)*h -> contributes to C11, C12
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (c-a)*(e+f) -> contributes to C22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (b-d)*(g+h) -> contributes to C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For each block-level term t (0..6) and each scalar-level term s (0..6),
    # we get one rank-1 term with index r = t*7 + s
    for t in range(7):
        block_A, block_B, block_C = strassen_terms[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_terms[s]
            r_idx = t * 7 + s

            # Build U contribution
            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U[a_row * 4 + a_col, r_idx] += block_coeff * scalar_coeff

            # Build V contribution
            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V[b_row * 4 + b_col, r_idx] += block_coeff * scalar_coeff

            # Build W contribution
            # W uses trace convention: W[k*4+i] where i=row of C, k=col of C
            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr  # i index of C
                    c_col = 2 * cj + sc  # k index of C (column of C)
                    w_idx = c_col * 4 + c_row  # W[k*4+i]
                    W[w_idx, r_idx] += block_coeff * scalar_coeff

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
