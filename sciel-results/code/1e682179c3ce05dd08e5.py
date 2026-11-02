"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition via Strassen's algorithm applied
to 2x2 block multiplications, exploiting the rectangular asymmetry (n=2).
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    - A is 2x4, B is 4x4, C is 2x4
    - Split A into left (cols 0-1) and right (cols 2-3) halves: A_L, A_R (each 2x2)
    - Split B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR
    - C[:, 0:2] = A_L @ B_TL + A_R @ B_BL
    - C[:, 2:4] = A_L @ B_TR + A_R @ B_BR
    
    Each 2x2x2 multiplication can be done with Strassen's 7 multiplications.
    We have 4 such multiplications, but we can share some intermediate values.
    
    Actually, let's use a more direct approach:
    For each of the 4 block multiplications (2x2 * 2x2 -> 2x2), use Strassen's 7 terms.
    Total rank = 4 * 7 = 28.
    
    Strassen's algorithm for 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms for 2x2x2 multiplication
    # Each term is (u_coeffs, v_coeffs, w_coeffs) where keys are local (row, col) indices
    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22)
        (
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): 1}
        ),
        # M2 = (A21 + A22)B11
        (
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 0): 1}
        ),
        # M3 = A11(B21 - B22)
        (
            {(0, 0): 1},
            {(1, 0): 1, (1, 1): -1},
            {(0, 1): 1, (1, 0): 1}
        ),
        # M4 = A22(B12 - B11)
        (
            {(1, 1): 1},
            {(0, 1): 1, (0, 0): -1},
            {(0, 0): 1, (1, 0): 1}
        ),
        # M5 = (A11 + A12)B22
        (
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): -1, (0, 1): 1}
        ),
        # M6 = (A21 - A11)(B11 + B12)
        (
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1}
        ),
        # M7 = (A12 - A22)(B21 + B22)
        (
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1, (1, 1): -1}
        ),
    ]
    
    # Define the 4 block multiplications
    # Each is: A_block (rows, cols), B_block (rows, cols), C_block (rows, cols)
    # All blocks are 2x2
    sub_mults = [
        # A_L @ B_TL -> C[:, 0:2]
        {
            'A': (0, 0),  # rows 0-1, cols 0-1
            'B': (0, 0),  # rows 0-1, cols 0-1
            'C': (0, 0)   # rows 0-1, cols 0-1
        },
        # A_R @ B_BL -> C[:, 0:2]
        {
            'A': (0, 2),  # rows 0-1, cols 2-3
            'B': (2, 0),  # rows 2-3, cols 0-1
            'C': (0, 0)   # rows 0-1, cols 0-1
        },
        # A_L @ B_TR -> C[:, 2:4]
        {
            'A': (0, 0),  # rows 0-1, cols 0-1
            'B': (0, 2),  # rows 0-1, cols 2-3
            'C': (0, 2)   # rows 0-1, cols 2-3
        },
        # A_R @ B_BR -> C[:, 2:4]
        {
            'A': (0, 2),  # rows 0-1, cols 2-3
            'B': (2, 2),  # rows 2-3, cols 2-3
            'C': (0, 2)   # rows 0-1, cols 2-3
        }
    ]
    
    col = 0
    for sub in sub_mults:
        A_r0, A_c0 = sub['A']
        B_r0, B_c0 = sub['B']
        C_r0, C_c0 = sub['C']
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for (ai, aj), coeff_u in u_coeffs.items():
                global_ai = A_r0 + ai
                global_aj = A_c0 + aj
                idx_u = global_ai * m + global_aj
                U[idx_u][col] += coeff_u
            
            for (bi, bj), coeff_v in v_coeffs.items():
                global_bi = B_r0 + bi
                global_bj = B_c0 + bj
                idx_v = global_bi * p + global_bj
                V[idx_v][col] += coeff_v
            
            for (ci, cj), coeff_w in w_coeffs.items():
                global_ci = C_r0 + ci
                global_cj = C_c0 + cj
                # Trace convention: W[k*n+i] where k=cj (output col), i=ci (output row)
                idx_w = global_cj * n + global_ci
                W[idx_w][col] += coeff_w
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
