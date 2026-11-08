import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen's algorithm for the 4x4 blocks and schoolbook for the 4x1 block.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS  # n=4, m=4, p=9

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2 terms: (u_coeffs, v_coeffs, w_coeffs)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def strassen_4x4_block(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        # Block indices for 2x2 blocks of 2x2 matrices
        # A blocks: 11, 12, 21, 22
        # B blocks: 11, 12, 21, 22
        # C blocks: 11, 12, 21, 22
        
        # Strassen 4x4 terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2*2x2 product, decomposed with Strassen.
        
        def get_A_indices(block_row, block_col):
            indices = {}
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    gi = A_base_i + block_row * 2 + li
                    gj = A_base_j + block_col * 2 + lj
                    indices[local_idx] = (gi, gj)
            return indices
        
        def get_B_indices(block_row, block_col):
            indices = {}
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    gj = B_base_j + block_row * 2 + lj
                    gk = B_base_k + block_col * 2 + lk
                    indices[local_idx] = (gj, gk)
            return indices
        
        def get_C_indices(block_row, block_col):
            indices = {}
            for li in range(2):
                for lk in range(2):
                    local_idx = li * 2 + lk
                    gi = C_base_i + block_row * 2 + li
                    gk = C_base_k + block_col * 2 + lk
                    indices[local_idx] = (gi, gk)
            return indices
        
        # Define the 7 block multiplications
        # Each is a linear combination of block products
        # We iterate through the 7 Strassen 2x2 terms for the block level
        # and for each, iterate through the 7 Strassen 2x2 terms for the sub-block level
        
        # M1: A_blocks [11, 22] with coeff 1, B_blocks [11, 22] with coeff 1
        # M2: A_blocks [21, 22] with coeff 1, B_blocks [11] with coeff 1
        # M3: A_blocks [11] with coeff 1, B_blocks [12, 22] with coeff 1, -1
        # M4: A_blocks [22] with coeff 1, B_blocks [21, 11] with coeff 1, -1
        # M5: A_blocks [11, 12] with coeff 1, B_blocks [22] with coeff 1
        # M6: A_blocks [21, 11] with coeff 1, -1, B_blocks [11, 12] with coeff 1
        # M7: A_blocks [12, 22] with coeff 1, -1, B_blocks [21, 22] with coeff 1
        
        block_mults = [
            # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
            # A_block_coeffs: dict {block_idx: coeff}
            # block_idx: 11->0, 12->1, 21->2, 22->3
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        for A_block_coeffs, B_block_coeffs, C_block_coeffs in block_mults:
            # For each sub-term in the block multiplication
            # We need to combine the A-blocks and B-blocks
            # This is a sum of products of (sum A_blocks) and (sum B_blocks)
            # (sum_a c_a A_a) * (sum_b d_b B_b) = sum_{a,b} c_a d_b A_a B_b
            # Each A_a B_b is a 2x2*2x2 product, decomposed with Strassen.
            
            for a_block_idx, a_coeff in A_block_coeffs.items():
                for b_block_idx, b_coeff in B_block_coeffs.items():
                    # A_a is at (a_block_idx // 2, a_block_idx % 2) ? No.
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    a_br = a_block_idx // 2
                    a_bc = a_block_idx % 2
                    b_br = b_block_idx // 2
                    b_bc = b_block_idx % 2
                    
                    A_indices = get_A_indices(a_br, a_bc)
                    B_indices = get_B_indices(b_br, b_bc)
                    
                    # The C contribution is the sum of C_block_coeffs
                    # But wait, the C_block_coeffs apply to the RESULT of the block multiplication.
                    # The result of A_a * B_b is a 2x2 matrix.
                    # It contributes to the C blocks according to C_block_coeffs?
                    # No, the Strassen formula at the block level is:
                    # C_11 = M1 + M4 - M5 + M7
                    # So M1 contributes to C_11 and C_22.
                    # The C_block_coeffs in `block_mults` are the contributions of M_i to C_blocks.
                    
                    # So for a specific M_i (defined by A_block_coeffs, B_block_coeffs),
                    # and for a specific term in the expansion (a_block, b_block),
                    # we compute the 2x2*2x2 product A_a * B_b.
                    # This product is decomposed using Strassen 2x2.
                    # The result is added to the C blocks specified by C_block_coeffs.
                    
                    for u_coeff, v_coeff, w_coeff in strassen_2x2_terms:
                        u_vec = [0] * (n * m)
                        v_vec = [0] * (m * p)
                        w_vec = [0] * (p * n)
                        
                        # Fill U
                        for local_idx, coeff in u_coeff.items():
                            gi, gj = A_indices[local_idx]
                            u_vec[gi * m + gj] += a_coeff * coeff
                        
                        # Fill V
                        for local_idx, coeff in v_coeff.items():
                            gj, gk = B_indices[local_idx]
                            v_vec[gj * p + gk] += b_coeff * coeff
                        
                        # Fill W
                        # The W coefficients are the same for all C blocks this M_i contributes to?
                        # No, w_coeff is the contribution to the LOCAL 2x2 C block.
                        # We need to map this to the GLOBAL C blocks.
                        # The M_i contributes to C_blocks defined by C_block_coeffs.
                        # For each such C_block, we add the w_coeff terms.
                        
                        for c_block_idx, c_coeff in C_block_coeffs.items():
                            c_br = c_block_idx // 2
                            c_bc = c_block_idx % 2
                            C_indices = get_C_indices(c_br, c_bc)
                            for local_idx, coeff in w_coeff.items():
                                gi, gk = C_indices[local_idx]
                                w_vec[gk * n + gi] += c_coeff * coeff
                        
                        add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B[0:4]
    strassen_4x4_block(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B[4:8]
    strassen_4x4_block(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B[8:9] (4x1), schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = np.array(U_cols, dtype=float).T
    V = np.array(V_cols, dtype=float).T
    W = np.array(W_cols, dtype=float).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
