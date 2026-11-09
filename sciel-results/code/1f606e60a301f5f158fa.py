"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-56 tensor decomposition for 2x4 @ 4x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Strassen 7 products for 2x2x2
    # Each product is defined by dictionaries mapping (row, col) to coefficient
    strassen_products = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    # 4 blocks of columns in B (size 2 each)
    # 2 blocks of rows in B / cols in A (size 2 each)
    for b_col_block in range(p // 2):
        for b_row_block in range(m // 2):
            # Offsets
            a_col_off = b_row_block * 2
            b_row_off = b_row_block * 2
            b_col_off = b_col_block * 2
            w_col_off = b_col_block * 2
            
            for u_dict, v_dict, w_dict in strassen_products:
                # Fill U
                for (i, j), val in u_dict.items():
                    idx_u = i * m + (a_col_off + j)
                    U[idx_u, col_idx] += val
                
                # Fill V
                for (j, k), val in v_dict.items():
                    idx_v = (b_row_off + j) * p + (b_col_off + k)
                    V[idx_v, col_idx] += val
                
                # Fill W
                for (k, i), val in w_dict.items():
                    idx_w = (w_col_off + k) * n + i
                    W[idx_w, col_idx] += val
                
                col_idx += 1
    
    # Ensure exact half-integers
    def clean_matrix(mat):
        out = np.zeros_like(mat)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                # Round to nearest 0.5
                rv = round(v * 2) / 2.0
                if abs(v - rv) < 1e-10:
                    out[r, c] = rv
                else:
                    # Fallback to fraction if not half-integer (should not happen for Strassen)
                    f = Fraction(v).limit_denominator(1000)
                    out[r, c] = float(f)
        return out.tolist()
        
    return {
        "U": clean_matrix(U),
        "V": clean_matrix(V),
        "W": clean_matrix(W)
    }
# EVOLVE-BLOCK-END
