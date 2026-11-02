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
    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen terms for 2x2 matrix multiplication
    # Each term is (A_dict, B_dict, C_dict)
    # Keys are (row, col) within the 2x2 block, values are coefficients
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Recursive Strassen: 
    # We treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The 7 Strassen products M1..M7 are computed using the 2x2 Strassen formulas.
    # However, the "scalars" in the 2x2 Strassen are themselves 2x2 matrices.
    # So we need to expand the 2x2 Strassen formula for each of the 7 products.
    
    # The 7 products M_t (t=0..6) are linear combinations of the block products.
    # Actually, it's easier to view the 49 terms as the tensor product of the two Strassen decompositions.
    # The rank is 7 * 7 = 49.
    
    # For each of the 7 outer Strassen terms (t), and 7 inner Strassen terms (s),
    # we have a bilinear form.
    
    for t in range(7):
        block_A, block_B, block_C = strassen_terms[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_terms[s]
            r_idx = t * 7 + s

            # Expand U
            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U[a_row * 4 + a_col, r_idx] += block_coeff * scalar_coeff

            # Expand V
            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V[b_row * 4 + b_col, r_idx] += block_coeff * scalar_coeff

            # Expand W
            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr
                    c_col = 2 * cj + sc
                    w_idx = c_col * 4 + c_row
                    W[w_idx, r_idx] += block_coeff * scalar_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
