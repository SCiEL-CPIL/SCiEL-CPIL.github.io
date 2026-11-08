import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 for 4x4x4 part, 16 for 4x4x1 part
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Multiplication (Rank 49) ---
    # We use Strassen's algorithm for 2x2 blocks.
    # Each 2x2 block multiplication has rank 7.
    # There are 7 such block multiplications in the outer Strassen step.
    # Total 7 * 7 = 49 terms.
    
    # Inner 2x2x2 Strassen decomposition vectors
    # U_in, V_in, W_in are 4x7 matrices (flattened 2x2)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_in = [
        [1, 0, 1, 0, 1, 0, 0], # 00
        [0, 1, 0, 0, 1, 1, 0], # 01
        [0, 1, 0, 0, 0, 1, 1], # 10
        [1, 1, 0, 1, 0, 0, -1]  # 11
    ]
    
    V_in = [
        [1, 1, 0, 0, 0, 1, 0], # 00
        [0, 0, 1, -1, 0, 1, 1], # 01
        [0, 0, 0, 1, 0, 0, 1], # 10
        [1, 0, -1, 0, 1, 0, 0]  # 11
    ]
    
    W_in = [
        [1, 0, 0, 0, 1, 1, 0], # 00
        [1, 0, 1, -1, 0, 1, 1], # 01
        [1, 1, 0, 1, 0, 0, 0], # 10
        [1, 1, -1, 0, 1, 0, 0]  # 11
    ]
    
    # Outer Strassen block combinations
    # Each entry: (A_blocks, B_blocks, C_block)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_block: block_idx (0-3)
    
    # Block indices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    outer_terms = [
        # M1 = (A00 + A11)(B00 + B11) -> C00
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], 0),
        # M2 = (A10 + A11) B00 -> C10
        ([ (2, 1), (3, 1) ], [ (0, 1) ], 2),
        # M3 = A00 (B01 - B11) -> C01
        ([ (0, 1) ], [ (1, 1), (3, -1) ], 1),
        # M4 = A11 (B10 - B00) -> C11
        ([ (3, 1) ], [ (2, 1), (0, -1) ], 3),
        # M5 = (A00 + A01) B11 -> C01
        ([ (0, 1), (1, 1) ], [ (3, 1) ], 1),
        # M6 = (A10 - A00) (B00 + B01) -> C10
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], 2),
        # M7 = (A01 - A11) (B10 + B11) -> C11
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], 3),
    ]
    
    col = 0
    for A_blocks, B_blocks, C_block in outer_terms:
        for k_in in range(7):
            # Get inner vectors
            u_vec = [U_in[i][k_in] for i in range(4)]
            v_vec = [V_in[i][k_in] for i in range(4)]
            w_vec = [W_in[i][k_in] for i in range(4)]
            
            # Expand U
            for i_in in range(4):
                if u_vec[i_in] == 0: continue
                # i_in maps to (r, c) in 2x2
                r_in, c_in = i_in // 2, i_in % 2
                # A_blocks tells us which global blocks to use
                for block_idx, coeff in A_blocks:
                    # block_idx 0: r 0-1, c 0-1
                    # block_idx 1: r 0-1, c 2-3
                    # block_idx 2: r 2-3, c 0-1
                    # block_idx 3: r 2-3, c 2-3
                    r_base = (block_idx // 2) * 2
                    c_base = (block_idx % 2) * 2
                    r_global = r_base + r_in
                    c_global = c_base + c_in
                    u_idx = r_global * m + c_global
                    U[u_idx][col] += u_vec[i_in] * coeff
            
            # Expand V
            for j_in in range(4):
                if v_vec[j_in] == 0: continue
                r_in, c_in = j_in // 2, j_in % 2
                for block_idx, coeff in B_blocks:
                    r_base = (block_idx // 2) * 2
                    c_base = (block_idx % 2) * 2
                    r_global = r_base + r_in
                    c_global = c_base + c_in
                    v_idx = r_global * p + c_global
                    V[v_idx][col] += v_vec[j_in] * coeff
            
            # Expand W
            for k_in_idx in range(4):
                if w_vec[k_in_idx] == 0: continue
                r_in, c_in = k_in_idx // 2, k_in_idx % 2
                # C_block is the target block
                r_base = (C_block // 2) * 2
                c_base = (C_block % 2) * 2
                r_global = r_base + r_in
                c_global = c_base + c_in
                w_idx = r_global * n + c_global
                W[w_idx][col] += w_vec[k_in_idx]
            
            col += 1
    
    # --- Part 2: 4x4x1 Multiplication (Rank 16) ---
    # Multiplies A (4x4) by B_col4 (4x1) to get C_col4 (4x1)
    # k=4
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
