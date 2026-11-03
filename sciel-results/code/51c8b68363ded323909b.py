import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # Block size b = 3.
    b = 3
    # Rank of 3x3 schoolbook is 27.
    # Strassen rank factor is 7.
    # Total rank = 7 * 27 = 189.
    
    rank_3x3 = b * b * b
    total_rank = 7 * rank_3x3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the linear combinations for A and B for each M
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # For each M, we have coefficients for A blocks and B blocks.
    # Then we use the 3x3 schoolbook decomposition for the product of the resulting 3x3 matrices.
    
    # 3x3 schoolbook:
    # U3[i*b+j, r] = 1 if r corresponds to (i,j,k)
    # V3[j*b+k, r] = 1
    # W3[k*b+i, r] = 1
    
    # We will iterate over the 7 Strassen terms.
    # For each term, we have a linear combination of A blocks and B blocks.
    # Let LA = {block_idx: coeff}
    # Let LB = {block_idx: coeff}
    
    # Correct mapping:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: A = A00+A11, B = B00+B11. Contributes to C00(+1), C11(+1)
    # M2: A = A10+A11, B = B00. Contributes to C10(+1), C11(-1)
    # M3: A = A00, B = B01-B11. Contributes to C01(+1), C11(+1)
    # M4: A = A11, B = B10-B00. Contributes to C00(+1), C10(+1)
    # M5: A = A00+A01, B = B11. Contributes to C00(-1), C01(+1)
    # M6: A = A10-A00, B = B00+B01. Contributes to C11(+1)
    # M7: A = A01-A11, B = B10+B11. Contributes to C00(+1)
    
    terms = [
        # (A_coeffs, B_coeffs, C_contribs)
        # A_coeffs: dict {(block_i, block_j): coeff}
        # B_coeffs: dict {(block_i, block_j): coeff}
        # C_contribs: list of ((block_i, block_j), coeff)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            [((0,0), 1), ((1,1), 1)]
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            [((1,0), 1), ((1,1), -1)]
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            [((0,1), 1), ((1,1), 1)]
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            [((0,0), 1), ((1,0), 1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            [((0,0), -1), ((0,1), 1)]
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            [((1,1), 1)]
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            [((0,0), 1)]
        ),
    ]
    
    for t_idx, (a_coeffs, b_coeffs, c_contribs) in enumerate(terms):
        base_rank = t_idx * rank_3x3
        # Iterate over all 3x3 schoolbook terms
        for i in range(b):
            for j in range(b):
                for k in range(b):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                        
                    r_local = i * b * b + j * b + k
                    r_global = base_rank + r_local
                    
                    # U contribution
                    # U[i_local * b + j_local, r]
                    # A is formed by sum of blocks.
                    # A_local[i, j] = sum_{(bi, bj)} coeff * A_block[bi, bj][i, j]
                    # So U for the 6x6 case:
                    # Row index in U is (I, J) where I = bi * b + i, J = bj * b + j.
                    # U[(bi*b+i)*6 + (bj*b+j), r] = coeff_a[bi,bj] * U3[i,j,r]
                    # U3[i,j,r] is 1 for schoolbook.
                    for (bi, bj), coeff_a in a_coeffs.items():
                        row_u = (bi * b + i) * m + (bj * b + j)
                        U[row_u][r_global] = coeff_a
                    
                    # V contribution
                    # V[j_local * b + k_local, r]
                    # B is formed by sum of blocks.
                    # B_local[j, k] = sum_{(bi, bj)} coeff * B_block[bi, bj][j, k]
                    # Row index in V is (J, K) where J = bi * b + j, K = bj * b + k.
                    # Note: Strassen B blocks are B00, B01, B10, B11.
                    # B00 is top-left.
                    # B_block[bi, bj] has rows bi*b..bi*b+b-1 and cols bj*b..bj*b+b-1.
                    # So V row is (bi*b+j)*p + (bj*b+k).
                    for (bi, bj), coeff_b in b_coeffs.items():
                        row_v = (bi * b + j) * p + (bj * b + k)
                        V[row_v][r_global] = coeff_b
                    
                    # W contribution
                    # W[k_local * b + i_local, r]
                    # C is sum of M terms.
                    # C_block[ci, cj] += coeff_c * M
                    # M is a 3x3 matrix.
                    # C_local[i, k] is the entry.
                    # C_block[ci, cj][i, k] = C[ci*b+i, cj*b+k]
                    # W row is (cj*b+k)*n + (ci*b+i).
                    for (ci, cj), coeff_c in c_contribs:
                        row_w = (cj * b + k) * n + (ci * b + i)
                        W[row_w][r_global] = coeff_c
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
