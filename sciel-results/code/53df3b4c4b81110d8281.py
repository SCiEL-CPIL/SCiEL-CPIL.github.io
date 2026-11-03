import numpy as np
import time

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a Strassen-based decomposition with rank 98.
    
    Approach: Split B (4x8) into two 4x4 blocks B1, B2.
    Compute C1 = A*B1 and C2 = A*B2 using 2-level Strassen (rank 49 each).
    Total rank: 98.
    """
    # Pre-computed Strassen factors for 2x2 matrix multiplication
    # Each factor is a tuple (u_vec, v_vec, w_vec) of length 4
    # u: A00, A01, A10, A11
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    S_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # P1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # P2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # P3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # P4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # P5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # P6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # P7
    ]
    
    # Pre-compute non-zero indices for 2x2 Strassen factors
    S_2x2_nz = []
    for u, v, w in S_2x2:
        S_2x2_nz.append((
            [(i, val) for i, val in enumerate(u) if val != 0],
            [(i, val) for i, val in enumerate(v) if val != 0],
            [(i, val) for i, val in enumerate(w) if val != 0]
        ))
        
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # For each of the 2 sub-problems (B1, B2)
    for t in range(2):
        # For each of the 7 top-level Strassen products
        for s in range(7):
            # For each of the 7 scalar Strassen products
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                u_top, v_top, w_top = S_2x2_nz[s]
                u_bot, v_bot, w_bot = S_2x2_nz[r]
                
                # Update U
                for i_block, i_val in u_top:
                    for i_sub, i_sub_val in u_bot:
                        i = 2 * i_block + i_sub
                        j_idx = 0 # Placeholder, need to map correctly
                        # Wait, u_top index i_block corresponds to 2x2 block index.
                        # u_top[i_block] is the coefficient for the block A_{i_block, j_block}?
                        # No, the Strassen factors are for the 2x2 blocks.
                        # The vector u has 4 entries: A00, A01, A10, A11.
                        # Index 0: A00 (block row 0, block col 0)
                        # Index 1: A01 (block row 0, block col 1)
                        # Index 2: A10 (block row 1, block col 0)
                        # Index 3: A11 (block row 1, block col 1)
                        
                        # So for u_top entry at index i_block (0..3):
                        # It corresponds to block row br = i_block // 2, block col bc = i_block % 2.
                        br = i_block // 2
                        bc = i_block % 2
                        
                        # The sub-vector u_bot has entries for the 2x2 matrix within that block.
                        # Index i_sub (0..3) corresponds to row er = i_sub // 2, col ec = i_sub % 2.
                        for i_sub, i_sub_val in u_bot:
                            er = i_sub // 2
                            ec = i_sub % 2
                            i = 2 * br + er
                            j = 2 * bc + ec
                            U[i*m + j, col] += i_val * i_sub_val

                # Update V
                for j_block, j_val in v_top:
                    br = j_block // 2
                    bc = j_block % 2
                    for j_sub, j_sub_val in v_bot:
                        er = j_sub // 2
                        ec = j_sub % 2
                        j = 2 * br + er
                        k = 4 * t + 2 * bc + ec
                        V[j*p + k, col] += j_val * j_sub_val

                # Update W
                for k_block, k_val in w_top:
                    br = k_block // 2
                    bc = k_block % 2
                    for k_sub, k_sub_val in w_bot:
                        er = k_sub // 2
                        ec = k_sub % 2
                        i = 2 * br + er
                        k = 4 * t + 2 * bc + ec
                        W[k*n + i, col] += k_val * k_sub_val

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
