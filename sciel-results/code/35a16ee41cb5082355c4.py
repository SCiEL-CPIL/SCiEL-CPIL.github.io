"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 7 products
    # Format: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: [A00, A01, A10, A11]
    # B_coeffs: [B00, B01, B10, B11]
    # C_coeffs: [C00, C01, C10, C11]
    
    strassen_prods = [
        ((0, 0, 0, 1), (1, 0, -1, 0), (1, 0, 0, 1)),   # M1
        ((1, 0, 0, 1), (0, 0, 0, 1), (1, 0, 0, -1)),   # M2
        ((1, 0, 0, 0), (0, 1, -1, 0), (0, 1, 0, 1)),   # M3
        ((0, 0, 0, 1), (1, 1, 0, 0), (-1, 1, 0, 0)),   # M4
        ((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)),    # M5
        ((1, -1, 0, 0), (1, 0, 1, 0), (0, 0, 1, 0)),   # M6
        ((0, 1, 0, 1), (-1, 0, 1, 0), (1, 0, 0, 0))    # M7
    ]
    
    # Define blocks
    # (a_row_off, a_col_off, b_row_off, b_col_off, c_col_off)
    blocks = [
        (0, 0, 0, 0, 0),  # A_L B_T0 -> C[:, 0:2]
        (0, 0, 0, 2, 2),  # A_L B_T1 -> C[:, 2:4]
        (0, 0, 0, 4, 4),  # A_L B_T2 -> C[:, 4:6]
        (0, 2, 2, 0, 0),  # A_R B_B0 -> C[:, 0:2]
        (0, 2, 2, 2, 2),  # A_R B_B1 -> C[:, 2:4]
        (0, 2, 2, 4, 4)   # A_R B_B2 -> C[:, 4:6]
    ]
    
    col = 0
    for b_off in blocks:
        a_ro, a_co, b_ro, b_co, c_co = b_off
        
        for prod in strassen_prods:
            a_coeffs, b_coeffs, c_coeffs = prod
            
            # Update U
            # A local indices: (0,0), (0,1), (1,0), (1,1)
            # Global A index: r * m + c
            U_idx_offsets = [
                (0 + a_ro) * m + (0 + a_co),
                (0 + a_ro) * m + (1 + a_co),
                (1 + a_ro) * m + (0 + a_co),
                (1 + a_ro) * m + (1 + a_co)
            ]
            for idx, coeff in zip(U_idx_offsets, a_coeffs):
                if coeff != 0:
                    U[idx][col] += coeff
                    
            # Update V
            # B local indices: (0,0), (0,1), (1,0), (1,1)
            # Global B index: r * p + c
            V_idx_offsets = [
                (0 + b_ro) * p + (0 + b_co),
                (0 + b_ro) * p + (1 + b_co),
                (1 + b_ro) * p + (0 + b_co),
                (1 + b_ro) * p + (1 + b_co)
            ]
            for idx, coeff in zip(V_idx_offsets, b_coeffs):
                if coeff != 0:
                    V[idx][col] += coeff
                    
            # Update W
            # C local indices: (0,0), (0,1), (1,0), (1,1)
            # Local (r, c) corresponds to Global C row r, col c + c_co
            # W index: c_global * n + r_global
            W_idx_offsets = [
                (0 + c_co) * n + 0,
                (1 + c_co) * n + 0,
                (0 + c_co) * n + 1,
                (1 + c_co) * n + 1
            ]
            for idx, coeff in zip(W_idx_offsets, c_coeffs):
                if coeff != 0:
                    W[idx][col] += coeff
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
