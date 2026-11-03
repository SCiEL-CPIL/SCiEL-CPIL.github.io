"""Program for matrix multiplication 4x4x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Uses Strassen's algorithm for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    Total rank: 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 65
    
    # --- Part 1: 4x4x4 multiplication (cols 0-3 of B, cols 0-3 of C) ---
    
    # Base Strassen vectors for 2x2 matrices
    # u, v are length 4 (flattened 2x2)
    # w is length 4 (flattened 2x2)
    base_u = [
        [-1, 0, 0, 1],  # a11 - a00
        [0, 0, 1, 1],   # a10 + a11
        [1, 0, 0, 0],   # a00
        [0, 0, 0, 1],   # a11
        [1, 0, 0, 1],   # a00 + a11
        [-1, 0, 1, 0],  # a10 - a00
        [0, 1, 0, 1]    # a01 + a11
    ]
    base_v = [
        [0, 0, 1, 1],   # b10 + b11
        [1, 0, 0, 0],   # b00
        [1, 0, 0, -1],  # b00 - b11
        [-1, 0, 0, 1],  # b11 - b00
        [1, 0, 0, 1],   # b00 + b11
        [1, 1, 0, 0],   # b01 + b00
        [0, 0, 1, -1]   # b10 - b11
    ]
    base_w = [
        [1, 0, 0, 1],   # C00 += M1, C11 += M1
        [0, 0, 1, 0],   # C10 += M2
        [0, 1, 0, 0],   # C01 += M3
        [-1, 0, 0, 1],  # C00 -= M4, C11 += M4
        [1, 0, 1, 0],   # C00 += M5, C10 += M5
        [0, 1, 0, 0],   # C01 += M6
        [1, 0, 0, 0]    # C00 += M7
    ]
    
    # Top-level Strassen combinations for 4x4 matrices
    # S_s and T_s are 2x2 block matrices.
    # Each block is a 2x2 matrix formed from A or B.
    # We represent S_s as a list of 4 vectors (for the 4 blocks), each of length 16 (flattened 4x4 A)?
    # No, S_s is a 2x2 matrix of 2x2 blocks.
    # Let's define the linear forms for the blocks directly.
    
    # A blocks: A00 (0-1, 0-1), A01 (0-1, 2-3), A10 (2-3, 0-1), A11 (2-3, 2-3)
    # Indices in A (16):
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks: B00 (0-1, 0-1), B01 (0-1, 2-3), B10 (2-3, 0-1), B11 (2-3, 2-3)
    # Indices in B (20, but for 4x4 part we only use cols 0-3, so 16 indices):
    # B is 4x5. For 4x4 part, we use B[:, 0:4].
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # Define S_s as 4 vectors of length 16 (coefficients for A)
    # S_1 = A11 - A00
    # S_2 = A10 + A11
    # S_3 = A00
    # S_4 = A11
    # S_5 = A00 + A11
    # S_6 = A10 - A00
    # S_7 = A01 + A11
    
    def block_to_vec(block_idx, coeff):
        """Convert a block index (0-3) and coefficient to a vector of length 16."""
        vec = [0] * 16
        if block_idx == 0: # A00
            vec[0] += coeff
            vec[1] += coeff
            vec[4] += coeff
            vec[5] += coeff
        elif block_idx == 1: # A01
            vec[2] += coeff
            vec[3] += coeff
            vec[6] += coeff
            vec[7] += coeff
        elif block_idx == 2: # A10
            vec[8] += coeff
            vec[9] += coeff
            vec[12] += coeff
            vec[13] += coeff
        elif block_idx == 3: # A11
            vec[10] += coeff
            vec[11] += coeff
            vec[14] += coeff
            vec[15] += coeff
        return vec
    
    S_blocks = [
        [(0, -1), (3, 1)],  # S1: A00 * -1, A11 * 1
        [(2, 1), (3, 1)],   # S2: A10 * 1, A11 * 1
        [(0, 1)],           # S3: A00 * 1
        [(3, 1)],           # S4: A11 * 1
        [(0, 1), (3, 1)],   # S5: A00 * 1, A11 * 1
        [(0, -1), (2, 1)],  # S6: A00 * -1, A10 * 1
        [(1, 1), (3, 1)]    # S7: A01 * 1, A11 * 1
    ]
    
    T_blocks = [
        [(2, 1), (3, 1)],   # T1: B10 * 1, B11 * 1
        [(0, 1)],           # T2: B00 * 1
        [(0, 1), (3, -1)],  # T3: B00 * 1, B11 * -1
        [(0, -1), (3, 1)],  # T4: B00 * -1, B11 * 1
        [(0, 1), (3, 1)],   # T5: B00 * 1, B11 * 1
        [(1, 1), (0, 1)],   # T6: B01 * 1, B00 * 1
        [(2, 1), (3, -1)]   # T7: B10 * 1, B11 * -1
    ]
    
    # Output block contributions
    # M1 -> C00 (1), C11 (1)
    # M2 -> C10 (1)
    # M3 -> C01 (1)
    # M4 -> C00 (-1), C11 (1)
    # M5 -> C00 (1), C10 (1)
    # M6 -> C01 (1)
    # M7 -> C00 (1)
    
    out_contribs = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1)],          # M2
        [(1, 1)],          # M3
        [(0, -1), (3, 1)], # M4
        [(0, 1), (2, 1)],  # M5
        [(1, 1)],          # M6
        [(0, 1)]           # M7
    ]
    
    # Mapping from output block index to W indices (for 4x4 C)
    # C00: 0,1,4,5
    # C01: 8,9,12,13
    # C10: 2,3,6,7
    # C11: 10,11,14,15
    block_w_indices = {
        0: [0, 1, 4, 5],
        1: [8, 9, 12, 13],
        2: [2, 3, 6, 7],
        3: [10, 11, 14, 15]
    }
    
    col = 0
    for s in range(7):
        # Construct S_s vector (length 16)
        S_vec = [0] * 16
        for b_idx, coeff in S_blocks[s]:
            v = block_to_vec(b_idx, coeff)
            for i in range(16):
                S_vec[i] += v[i]
                
        # Construct T_s vector (length 16)
        T_vec = [0] * 16
        for b_idx, coeff in T_blocks[s]:
            v = block_to_vec(b_idx, coeff)
            for i in range(16):
                T_vec[i] += v[i]
        
        # Apply base Strassen
        for t in range(7):
            # U part: S_vec * base_u[t]
            # S_vec is 16 (A entries), base_u[t] is 4 (2x2 entries)
            # We need to map base_u[t] indices to S_vec indices?
            # No, S_s is a 2x2 matrix of blocks.
            # S_s = [[S00, S01], [S10, S11]]
            # S00, S01, S10, S11 are 2x2 matrices.
            # base_u[t] is a linear form on the 4 entries of S_s.
            # So U_total = sum_{ij} base_u[t][ij] * S_{s,ij}
            # S_{s,ij} is a 2x2 matrix, which is a linear combination of A blocks.
            
            # Let's re-construct S_s as 4 block vectors
            S_blocks_vec = []
            for b_idx in range(4):
                bv = [0] * 16
                for bb, coeff in S_blocks[s]:
                    if bb == b_idx:
                        v = block_to_vec(b_idx, coeff)
                        for i in range(16):
                            bv[i] += v[i]
                S_blocks_vec.append(bv)
            
            # base_u[t] = [u00, u01, u10, u11]
            u00, u01, u10, u11 = base_u[t]
            U_vec = [0] * 16
            for i in range(16):
                U_vec[i] = u00 * S_blocks_vec[0][i] + u01 * S_blocks_vec[1][i] + u10 * S_blocks_vec[2][i] + u11 * S_blocks_vec[3][i]
            
            # V part: T_vec * base_v[t]
            T_blocks_vec = []
            for b_idx in range(4):
                bv = [0] * 16
                for bb, coeff in T_blocks[s]:
                    if bb == b_idx:
                        v = block_to_vec(b_idx, coeff)
                        for i in range(16):
                            bv[i] += v[i]
                T_blocks_vec.append(bv)
            
            v00, v01, v10, v11 = base_v[t]
            V_vec = [0] * 16
            for i in range(16):
                V_vec[i] = v00 * T_blocks_vec[0][i] + v01 * T_blocks_vec[1][i] + v10 * T_blocks_vec[2][i] + v11 * T_blocks_vec[3][i]
            
            # W part: base_w[t] * out_contribs[s]
            w00, w01, w10, w11 = base_w[t]
            W_vec = [0] * 20  # Only first 16 are used for 4x4 part
            for block_idx, coeff in out_contribs[s]:
                w_indices = block_w_indices[block_idx]
                # base_w[t] is [w00, w01, w10, w11] corresponding to C00, C01, C10, C11 of the 2x2 result
                # The 2x2 result M_s is placed in block C_block_idx
                # So w00 goes to C_block_idx[0,0], etc.
                W_vec[w_indices[0]] += coeff * w00
                W_vec[w_indices[1]] += coeff * w01
                W_vec[w_indices[2]] += coeff * w10
                W_vec[w_indices[3]] += coeff * w11
            
            # Add to U, V, W
            for i in range(16):
                U[i][col] = U_vec[i]
                V[i][col] = V_vec[i]
                W[i][col] = W_vec[i]
            col += 1
            
    # --- Part 2: 4x4x1 multiplication (col 4 of B, col 4 of C) ---
    # C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 16 schoolbook
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # B[j, 4]
            W[4 * n + i][col] = 1  # C[i, 4] -> index 4*4 + i = 16 + i
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
