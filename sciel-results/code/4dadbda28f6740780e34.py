"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """
    Builds the rank-28 Strassen decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Use integer dtype as coefficients are integers
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms for 2x2 * 2x2
    U_terms = [
        {0: 1, 3: 1},       
        {2: 1, 3: 1},       
        {0: 1},             
        {3: 1},             
        {0: 1, 1: 1},       
        {2: 1, 0: -1},      
        {1: 1, 3: -1},      
    ]
    
    V_terms = [
        {0: 1, 3: 1},       
        {0: 1},             
        {1: 1, 3: -1},      
        {2: 1, 0: -1},      
        {3: 1},             
        {0: 1, 1: 1},       
        {2: 1, 3: 1},       
    ]
    
    W_terms = [
        {0: 1, 3: 1},       
        {2: 1, 3: -1},      
        {1: 1, 3: 1},       
        {0: 1, 2: 1},       
        {0: -1, 1: 1},      
        {3: 1},             
        {0: 1},             
    ]
    
    blocks = [
        (0, 0, 0, 0), 
        (1, 1, 0, 0), 
        (0, 0, 1, 1), 
        (1, 1, 1, 1), 
    ]
    
    col_idx = 0
    for (a_off, b_row_off, b_col_off, c_col_off) in blocks:
        for s in range(7):
            for loc_idx, coeff in U_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_off * 2 + c
                U[g_r * 4 + g_c, col_idx] += coeff
            
            for loc_idx, coeff in V_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off * 2 + r
                g_c = b_col_off * 2 + c
                V[g_r * 4 + g_c, col_idx] += coeff
            
            for loc_idx, coeff in W_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off * 2 + c
                W[g_c * 2 + g_r, col_idx] += coeff
            
            col_idx += 1
            
    return U, V, W

def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _build_strassen_rank28()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
