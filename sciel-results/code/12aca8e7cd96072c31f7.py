import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 98 (2 * 49)
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 micro-terms for 2x2x2 Strassen
    # Each is (u_vec, v_vec, w_vec) of length 4
    # Indices: 00, 01, 10, 11
    micro_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Define the 7 macro-terms for 4x4x4 Strassen
    # Each is (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_idx, sign)
    # block_idx: 00, 01, 10, 11
    macro_terms = [
        # M1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]),
        # M3
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1) ]),
        # M4
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        # M5
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M7
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
    ]
    
    # Helper to get global indices for a 2x2 block
    def get_block_indices(block_idx, offset_j=0, offset_k=0):
        # block_idx: 0, 1, 2, 3 corresponding to 00, 01, 10, 11
        # Returns list of (i, j) or (j, k) or (k, i) depending on context
        # Here we just return the base offsets
        # For A (i, j):
        # 00: i in 0-1, j in 0-1
        # 01: i in 0-1, j in 2-3
        # 10: i in 2-3, j in 0-1
        # 11: i in 2-3, j in 2-3
        pass

    rank_idx = 0
    for b in range(2): # Two 4x4 blocks in the 4x8 matrix
        # k offset for V and W
        k_offset = 4 * b
        
        for m_idx in range(7):
            A_blocks, B_blocks, C_blocks = macro_terms[m_idx]
            
            for s_idx in range(7):
                u_vec, v_vec, w_vec = micro_terms[s_idx]
                
                # Process U
                for l in range(4):
                    val = u_vec[l]
                    if val != 0:
                        # l is local index in the 2x2 sum of blocks
                        # We need to map this to the global A indices
                        # The macro term defines which blocks are summed.
                        # Wait, the micro term operates on the 2x2 matrix A' which is the sum of blocks.
                        # So A'_{ij} = sum_{block in A_blocks} sign * A_{block, ij}
                        # The micro term uses A'_{l} (flattened).
                        # So we add val * sign * A_{block, l} to U.
                        
                        # Determine local (r, c) from l
                        r, c = l // 2, l % 2
                        
                        for block_idx, sign in A_blocks:
                            # Calculate global i, j
                            # block 0: i 0-1, j 0-1
                            # block 1: i 0-1, j 2-3
                            # block 2: i 2-3, j 0-1
                            # block 3: i 2-3, j 2-3
                            
                            i_base = (block_idx // 2) * 2
                            j_base = (block_idx % 2) * 2
                            
                            i_glob = i_base + r
                            j_glob = j_base + c
                            
                            U[i_glob * m + j_glob][rank_idx] += val * sign
                
                # Process V
                for l in range(4):
                    val = v_vec[l]
                    if val != 0:
                        r, c = l // 2, l % 2
                        
                        for block_idx, sign in B_blocks:
                            # B blocks are 2x2 submatrices of the 4x4 B block
                            # j is row, k is col
                            j_base = (block_idx // 2) * 2
                            k_base_local = (block_idx % 2) * 2
                            
                            j_glob = j_base + r
                            k_glob = k_offset + k_base_local + c
                            
                            V[j_glob * p + k_glob][rank_idx] += val * sign
                
                # Process W
                for l in range(4):
                    val = w_vec[l]
                    if val != 0:
                        r, c = l // 2, l % 2
                        
                        for block_idx, sign in C_blocks:
                            # C blocks are 2x2 submatrices of the 4x4 C block
                            # k is row, i is col
                            k_base_local = (block_idx // 2) * 2
                            i_base = (block_idx % 2) * 2
                            
                            k_glob = k_offset + k_base_local + r
                            i_glob = i_base + c
                            
                            W[k_glob * n + i_glob][rank_idx] += val * sign
                
                rank_idx += 1

    return {"U": U, "V": V, "W": W}
