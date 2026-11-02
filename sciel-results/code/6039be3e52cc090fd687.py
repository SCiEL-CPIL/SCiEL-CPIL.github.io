import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 multiplications per block = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(r, c): coeff} for A_local
    # v_coeffs: dict {(r, c): coeff} for B_local
    # w_coeffs: dict {(r, c): coeff} for X_local
    
    # M1 = (A00+A11)(B00+B11)
    term_0 = ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2 = (A10+A11)B00
    term_1 = ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
    # M3 = A00(B10-B11)
    term_2 = ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4 = A11(B01-B00)
    term_3 = ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1})
    # M5 = (A00+A01)B11
    term_4 = ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6 = (A10-A00)(B00+B01)
    term_5 = ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # M7 = (A01-A11)(B10+B11)
    term_6 = ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    
    terms = [term_0, term_1, term_2, term_3, term_4, term_5, term_6]
    
    # Block definitions
    # block: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # Note: A is 2x4, B is 4x4, C is 2x4
    # A_local(r,c) corresponds to A_global(r+a_row, c+a_col)
    # B_local(r,c) corresponds to B_global(r+b_row, c+b_col)
    # X_local(r,c) corresponds to C_global(r, c+c_col)
    # W index is k*n + i. Here k is the column of C, i is the row of C.
    # So W index for C_global(r, k) is k*n + r.
    
    blocks = [
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0, 0),
        # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 2, 2),
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0, 0),
        # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2, 2)
    ]
    
    for b_idx, (a_ro, a_co, b_ro, b_co, c_co) in enumerate(blocks):
        for t_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
            col = b_idx * 7 + t_idx
            
            # Fill U
            for (r, c), coeff in u_coeffs.items():
                global_r = r + a_ro
                global_c = c + a_co
                u_idx = global_r * m + global_c
                U[u_idx][col] += coeff
                
            # Fill V
            for (r, c), coeff in v_coeffs.items():
                global_r = r + b_ro
                global_c = c + b_co
                v_idx = global_r * p + global_c
                V[v_idx][col] += coeff
                
            # Fill W
            for (r, c), coeff in w_coeffs.items():
                global_r = r
                global_c = c + c_co
                # W index convention: W[k*n + i, r] where k is col, i is row
                # Here r is row i, c is col k
                w_idx = global_c * n + global_r
                W[w_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
