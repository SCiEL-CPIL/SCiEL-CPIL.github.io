import numpy as np

def construct():
    # Strassen terms for 2x2 * 2x2
    # Local indices for A, B, C (2x2, col-major)
    # A: 0:11, 1:21, 2:12, 3:22
    # B: 0:11, 1:21, 2:12, 3:22
    # C: 0:11, 1:21, 2:12, 3:22
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]), # M6
        ([0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 0, 1])  # M7
    ]
    
    rank = 56
    U = np.zeros((8, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Block definitions
    # A blocks: Left (cols 0,1), Right (cols 2,3)
    # A indices (col-major): 0:10, 1:20, 2:11, 3:21, 4:12, 5:22, 6:13, 7:23
    # Local A indices for 2x2: 0:11, 1:21, 2:12, 3:22
    # Left block maps local [0,1,2,3] to global [0,1,2,3]
    # Right block maps local [0,1,2,3] to global [4,5,6,7]
    
    a_mappings = [
        [0, 1, 2, 3], # Left
        [4, 5, 6, 7]  # Right
    ]
    
    # B blocks: Top (rows 0,1), Bottom (rows 2,3)
    # B is 4x8. Col-major index: k*4 + j.
    # Top blocks: j in {0,1}. k in {0,1}, {2,3}, {4,5}, {6,7}
    # Bottom blocks: j in {2,3}. k in {0,1}, {2,3}, {4,5}, {6,7}
    
    b_mappings = []
    for j_offset in [0, 2]: # 0 for Top, 2 for Bottom
        for k_start in [0, 2, 4, 6]:
            # Local B indices: 0:11, 1:21, 2:12, 3:22 (relative to 2x2 block)
            # Global indices:
            # 0: B[j_offset, k_start]
            # 1: B[j_offset+1, k_start]
            # 2: B[j_offset, k_start+1]
            # 3: B[j_offset+1, k_start+1]
            
            idx_0 = k_start * 4 + j_offset
            idx_1 = k_start * 4 + (j_offset + 1)
            idx_2 = (k_start + 1) * 4 + j_offset
            idx_3 = (k_start + 1) * 4 + (j_offset + 1)
            
            b_mappings.append([idx_0, idx_1, idx_2, idx_3])
            
    # Order of b_mappings:
    # 0: Top, k=0,1
    # 1: Top, k=2,3
    # 2: Top, k=4,5
    # 3: Top, k=6,7
    # 4: Bot, k=0,1
    # 5: Bot, k=2,3
    # 6: Bot, k=4,5
    # 7: Bot, k=6,7
    
    # C blocks: 2x8. Col-major index: k*2 + i.
    # 4 blocks of 2x2.
    c_mappings = []
    for k_start in [0, 2, 4, 6]:
        # Local C indices: 0:11, 1:21, 2:12, 3:22
        # Global:
        # 0: C[0, k_start]
        # 1: C[1, k_start]
        # 2: C[0, k_start+1]
        # 3: C[1, k_start+1]
        
        idx_0 = k_start * 2 + 0
        idx_1 = k_start * 2 + 1
        idx_2 = (k_start + 1) * 2 + 0
        idx_3 = (k_start + 1) * 2 + 1
        
        c_mappings.append([idx_0, idx_1, idx_2, idx_3])
        
    # Fill U, V, W
    # 8 blocks total (2 A-blocks * 4 B-blocks? No.)
    # We have 2 A-blocks (Left, Right).
    # We have 4 B-Top blocks and 4 B-Bottom blocks.
    # Left A multiplies Top B blocks.
    # Right A multiplies Bottom B blocks.
    
    block_idx = 0
    # Left A (a_mappings[0]) with Top B (b_mappings[0..3])
    for i in range(4):
        a_map = a_mappings[0]
        b_map = b_mappings[i]
        c_map = c_mappings[i]
        
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            for l in range(4):
                U[a_map[l], block_idx] = u_loc[l]
                V[b_map[l], block_idx] = v_loc[l]
                W[c_map[l], block_idx] = w_loc[l]
                
            block_idx += 1
            
    # Right A (a_mappings[1]) with Bottom B (b_mappings[4..7])
    for i in range(4):
        a_map = a_mappings[1]
        b_map = b_mappings[4 + i]
        c_map = c_mappings[i] # C output is the same blocks
        
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            for l in range(4):
                U[a_map[l], block_idx] = u_loc[l]
                V[b_map[l], block_idx] = v_loc[l]
                W[c_map[l], block_idx] = w_loc[l]
                
            block_idx += 1
            
    return {'U': U, 'V': V, 'W': W}
