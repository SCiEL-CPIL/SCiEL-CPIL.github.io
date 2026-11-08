import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # M1: (a00+a11)(b00+b11) -> C00+1, C11+1
    # M2: (a10+a11)b00 -> C10+1, C11-1
    # M3: a00(b01-b11) -> C01+1, C11+1
    # M4: a11(b10-b00) -> C00+1, C10+1
    # M5: (a00+a01)b11 -> C00-1, C01+1
    # M6: (a10-a00)(b00+b01) -> C11+1
    # M7: (a01-a11)(b10+b11) -> C00+1
    
    def add_block_strassen(base_rank, a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: top-left index of A block
        # b_base_i, b_base_j: top-left index of B block
        # c_base_i, c_base_j: top-left index of C block
        
        # A elements
        a00 = (a_base_i, a_base_j)
        a01 = (a_base_i, a_base_j + 1)
        a10 = (a_base_i + 1, a_base_j)
        a11 = (a_base_i + 1, a_base_j + 1)
        
        # B elements
        b00 = (b_base_i, b_base_j)
        b01 = (b_base_i, b_base_j + 1)
        b10 = (b_base_i + 1, b_base_j)
        b11 = (b_base_i + 1, b_base_j + 1)
        
        # C elements (target indices)
        c00 = (c_base_i, c_base_j)
        c01 = (c_base_i, c_base_j + 1)
        c10 = (c_base_i + 1, c_base_j)
        c11 = (c_base_i + 1, c_base_j + 1)
        
        # 7 Strassen terms
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: (i, j, coeff), V: (j, k, coeff), W: (i, k, coeff) -> mapped to W[k*4+i]
            
            # M1
            ([a00, a11], [b00, b11], [(c00, 1.0), (c11, 1.0)]),
            # M2
            ([a10, a11], [b00], [(c10, 1.0), (c11, -1.0)]),
            # M3
            ([a00], [b01, b11], [(c01, 1.0), (c11, 1.0)]), # b01 - b11
            # M4
            ([a11], [b10, b00], [(c00, 1.0), (c10, 1.0)]), # b10 - b00
            # M5
            ([a00, a01], [b11], [(c00, -1.0), (c01, 1.0)]),
            # M6
            ([a10, a00], [b00, b01], [(c11, 1.0)]), # a10 - a00, b00 + b01
            # M7
            ([a01, a11], [b10, b11], [(c00, 1.0)]) # a01 - a11, b10 + b11
        ]
        
        # Fix signs for subtraction in M3, M4, M6, M7
        # M3: a00 * (b01 - b11). V coeffs: b01:1, b11:-1
        # M4: a11 * (b10 - b00). V coeffs: b10:1, b00:-1
        # M6: (a10 - a00) * (b00 + b01). U coeffs: a10:1, a00:-1. V coeffs: b00:1, b01:1
        # M7: (a01 - a11) * (b10 + b11). U coeffs: a01:1, a11:-1. V coeffs: b10:1, b11:1
        
        # Re-define with explicit coeffs
        terms_corrected = [
            # M1
            ([(a00[0], a00[1], 1.0), (a11[0], a11[1], 1.0)], 
             [(b00[0], b00[1], 1.0), (b11[0], b11[1], 1.0)], 
             [(c00[0], c00[1], 1.0), (c11[0], c11[1], 1.0)]),
             # M2
            ([(a10[0], a10[1], 1.0), (a11[0], a11[1], 1.0)], 
             [(b00[0], b00[1], 1.0)], 
             [(c10[0], c10[1], 1.0), (c11[0], c11[1], -1.0)]),
             # M3
            ([(a00[0], a00[1], 1.0)], 
             [(b01[0], b01[1], 1.0), (b11[0], b11[1], -1.0)], 
             [(c01[0], c01[1], 1.0), (c11[0], c11[1], 1.0)]),
             # M4
            ([(a11[0], a11[1], 1.0)], 
             [(b10[0], b10[1], 1.0), (b00[0], b00[1], -1.0)], 
             [(c00[0], c00[1], 1.0), (c10[0], c10[1], 1.0)]),
             # M5
            ([(a00[0], a00[1], 1.0), (a01[0], a01[1], 1.0)], 
             [(b11[0], b11[1], 1.0)], 
             [(c00[0], c00[1], -1.0), (c01[0], c01[1], 1.0)]),
             # M6
            ([(a10[0], a10[1], 1.0), (a00[0], a00[1], -1.0)], 
             [(b00[0], b00[1], 1.0), (b01[0], b01[1], 1.0)], 
             [(c11[0], c11[1], 1.0)]),
             # M7
            ([(a01[0], a01[1], 1.0), (a11[0], a11[1], -1.0)], 
             [(b10[0], b10[1], 1.0), (b11[0], b11[1], 1.0)], 
             [(c00[0], c00[1], 1.0)])
        ]
        
        for k in range(7):
            r_idx = base_rank + k
            u_coeffs, v_coeffs, w_coeffs = terms_corrected[k]
            
            for (i, j, coeff) in u_coeffs:
                U[i * 4 + j][r_idx] += coeff
            for (j, k_idx, coeff) in v_coeffs:
                V[j * 4 + k_idx][r_idx] += coeff
            for (i, k_idx, coeff) in w_coeffs:
                # W index is k*4 + i
                W[k_idx * 4 + i][r_idx] += coeff

    # Block 00: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_block_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 01: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A rows 0,1 cols 2,3. B rows 2,3 cols 0,1.
    add_block_strassen(7, 0, 2, 2, 0, 0, 0)
    
    # Block 10: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    # A rows 2,3 cols 0,1. B rows 0,1 cols 2,3.
    add_block_strassen(14, 2, 0, 0, 2, 2, 2)
    
    # Block 11: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    # A rows 2,3 cols 2,3. B rows 2,3 cols 2,3.
    add_block_strassen(21, 2, 2, 2, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
