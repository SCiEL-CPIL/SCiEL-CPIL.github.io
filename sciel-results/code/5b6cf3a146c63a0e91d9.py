import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 9
    
    # Target: Find rank for 2x3x3, then scale
    # If we find rank 8 for 2x3x3, total rank is 24.
    # Else, use schoolbook 54.
    
    # 1. Try to find rank 8 decomposition for 2x3x3
    res_233 = find_rank_8_233()
    
    if res_233 is not None:
        # Construct 2x3x9 by tiling
        # U_233 is 6x8, V_233 is 9x8, W_233 is 6x8
        # For 2x3x9:
        # U is 6x24
        # V is 27x24
        # W is 18x24
        
        U_233, V_233, W_233 = res_233
        r_block = U_233.shape[1]
        total_rank = 3 * r_block
        
        U = np.zeros((6, total_rank))
        V = np.zeros((27, total_rank))
        W = np.zeros((18, total_rank))
        
        for k in range(3):
            # Block k uses columns k*r_block to (k+1)*r_block
            U[:, k*r_block:(k+1)*r_block] = U_233
            # V for block k: rows k*3 to k*3+2 (since B is 3x9, split into 3x3 blocks)
            # V_233 rows correspond to B indices (0,1,2) x (0,1,2) -> 0..8
            # In V_233, row index is j*p_block + k_block?
            # For 2x3x3, V shape is 9x8. Rows are (j, k_block) with j in 0..2, k_block in 0..2.
            # In 2x3x9, V shape is 27x24. Rows are (j, k) with j in 0..2, k in 0..8.
            # Block 0 uses k in 0..2.
            # So V_233 rows map to V rows 0..8.
            V[k*3*3:(k+1)*3*3, k*r_block:(k+1)*r_block] = V_233
            # W for block k: W_233 shape 6x8. Rows are (k_block, i) with k_block in 0..2, i in 0..1.
            # In 2x3x9, W shape 18x24. Rows are (k, i) with k in 0..8, i in 0..1.
            # Block 0 uses k in 0..2.
            W[k*2:(k+1)*2, k*r_block:(k+1)*r_block] = W_233
            
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

def find_rank_8_233():
    # Search for rank 8 decomposition of 2x3x3
    # Returns (U, V, W) or None
    pass
