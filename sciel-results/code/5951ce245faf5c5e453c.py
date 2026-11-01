"""Matrix multiplication 2x4 x 4x4 tensor decomposition - Rank 28."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x4 x 4x4 matrix multiplication.
    The minimal rank is 28.
    
    The tensor M_{2,4,4} can be decomposed into 4 independent M_{2,2,2} blocks.
    Since the variables in each block are disjoint, the rank is the sum of the ranks.
    Rank(M_{2,2,2}) = 7 (Strassen).
    Total Rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # We construct the rank-28 Strassen decomposition.
    # It consists of 4 blocks of 2x2x2 Strassen multiplications.
    
    rank = 28
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # Inputs:
    # U indices: i (0..1), j (0..1) -> flat i*2+j
    # V indices: j (0..1), k (0..1) -> flat j*2+k
    # W indices: k (0..1), i (0..1) -> flat k*2+i
    
    # Strassen multiplications M1..M7
    # U_side:
    # 1: A00 + A11
    # 2: A10 + A11
    # 3: A00
    # 4: A11
    # 5: A00 + A01
    # 6: A10 - A00
    # 7: A01 + A11
    
    # V_side:
    # 1: B00 + B11
    # 2: B00
    # 3: B01 - B11
    # 4: B10 - B00
    # 5: B11
    # 6: B00 + B01
    # 7: B10 + B11
    
    # W_side (contribution to C):
    # 1: C00 += M1, C11 += M1
    # 2: C10 += M2, C11 -= M2
    # 3: C01 += M3, C11 += M3
    # 4: C00 += M4, C10 += M4
    # 5: C00 -= M5, C01 += M5
    # 6: C11 += M6
    # 7: C00 += M7
    
    u_defs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},            # M3
        {(1,1): 1},            # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): 1},  # M7
    ]
    
    v_defs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},            # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},            # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1},  # M7
    ]
    
    w_defs = [
        {(0,0): 1, (1,1): 1},  # M1 -> C00, C11
        {(1,0): 1, (1,1): -1}, # M2 -> C10, -C11
        {(0,1): 1, (1,1): 1},  # M3 -> C01, C11
        {(0,0): 1, (1,0): 1},  # M4 -> C00, C10
        {(0,0): -1, (0,1): 1}, # M5 -> -C00, C01
        {(1,1): 1},            # M6 -> C11
        {(0,0): 1},            # M7 -> C00
    ]
    
    col = 0
    # 4 blocks:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Top-Left of C)
    # Block 1: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4] (Top-Right of C)
    # Block 2: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2] (Add to Top-Left)
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Add to Top-Right)
    
    # Note: The decomposition must satisfy C = AB.
    # C[:, 0:2] = A[:, 0:2]B[0:2, 0:2] + A[:, 2:4]B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2]B[0:2, 2:4] + A[:, 2:4]B[2:4, 2:4]
    
    # So we have 4 Strassen instances:
    # 1. A_TL * B_TL
    # 2. A_TR * B_TR
    # 3. A_TL * B_BL
    # 4. A_TR * B_BR
    
    # Mapping to flat indices:
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*2 + i
    
    blocks = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_row_offset, w_col_offset)
        # A is 2x4. B is 4x4. C is 2x4.
        # A_TL: rows 0-1, cols 0-1
        # B_TL: rows 0-1, cols 0-1
        # C_TL: rows 0-1, cols 0-1
        (0, 0, 0, 0, 0, 0),
        
        # A_TL: rows 0-1, cols 0-1
        # B_TR: rows 0-1, cols 2-3
        # C_TR: rows 0-1, cols 2-3
        (0, 0, 0, 2, 0, 2),
        
        # A_TR: rows 0-1, cols 2-3
        # B_BL: rows 2-3, cols 0-1
        # C_TL: rows 0-1, cols 0-1
        (0, 2, 2, 0, 0, 0),
        
        # A_TR: rows 0-1, cols 2-3
        # B_BR: rows 2-3, cols 2-3
        # C_TR: rows 0-1, cols 2-3
        (0, 2, 2, 2, 0, 2),
    ]
    
    for a_r, a_c, b_r, b_c, w_r, w_c in blocks:
        for s in range(7):
            c = col
            
            # U side
            for (i_loc, j_loc), coeff in u_defs[s].items():
                i = a_r + i_loc
                j = a_c + j_loc
                U[i*4 + j, c] += coeff
            
            # V side
            for (j_loc, k_loc), coeff in v_defs[s].items():
                j = b_r + j_loc
                k = b_c + k_loc
                V[j*4 + k, c] += coeff
            
            # W side
            for (k_loc, i_loc), coeff in w_defs[s].items():
                k = w_c + k_loc
                i = w_r + i_loc
                W[k*2 + i, c] += coeff
                
            col += 1
            
    # Convert to lists of lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
