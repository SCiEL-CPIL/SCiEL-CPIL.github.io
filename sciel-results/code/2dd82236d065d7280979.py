import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 Strassen
    # We need the 49 rank-1 terms for 4x4x4.
    # We generate them by applying Strassen to 2x2 blocks,
    # where each 2x2 block multiplication is done with 7 rank-1 terms.
    
    # First, define the 7 rank-1 terms for 2x2x2.
    # U, V, W for 2x2x2.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    
    # Strassen 2x2x2 vectors
    # We can derive them or use a known list.
    # Let's use the standard Strassen coefficients.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of u_r v_r w_r.
    # This is a bit tedious to hardcode directly for 2x2x2.
    # Instead, we can simulate the Strassen algorithm for 4x4x4
    # by treating the 2x2 blocks as "scalars" that are themselves rank-1 decomposable.
    
    # Let's define a helper to add a 2x2x2 decomposition to the main tensors.
    # The 2x2x2 decomposition has 7 terms.
    # We need the explicit 7 vectors for 2x2x2.
    
    # Here are the 7 vectors for 2x2x2 (u, v, w)
    # u, v are in R^4 (flattened 2x2), w is in R^4 (flattened 2x2 output).
    # Actually, for 2x2x2, U is 4x7, V is 4x7, W is 4x7.
    
    # Let's generate the 7 vectors for 2x2x2 by solving the linear system
    # or using the known Strassen decomposition.
    
    # Known Strassen 2x2x2 vectors (u, v, w):
    # 1. u=[1,0,1,0], v=[1,0,0,1], w=[1,0,0,1]  # M1
    # 2. u=[0,1,1,0], v=[1,0,0,0], w=[0,0,1,0]  # M2
    # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] # M3
    # 4. u=[0,0,1,0], v=[0,1,-1,0], w=[0,0,1,0] # M4
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]  # M5
    # 6. u=[0,1,-1,0], v=[1,1,0,0], w=[0,0,1,0] # M6
    # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[0,0,0,1] # M7
    
    # Let's verify M1:
    # u=[1,0,1,0] -> A00, A11
    # v=[1,0,0,1] -> B00, B11
    # w=[1,0,0,1] -> C00, C11
    # Product: A00*B00 -> C00, A00*B11 -> C11, A11*B00 -> C00, A11*B11 -> C11.
    # Sum: C00 gets A00B00+A11B00. C11 gets A00B11+A11B11.
    # This matches M1 = (A00+A11)(B00+B11).
    
    # We will use these 7 vectors for the 2x2x2 blocks.
    
    def add_2x2x2(offset_u, offset_v, offset_w, scale):
        nonlocal col
        # offset_u, offset_v, offset_w are the starting indices in the 4x4 blocks.
        # The 2x2 blocks are at (0,0), (0,1), (1,0), (1,1) in the 4x4 grid.
        # We need to map the local 2x2 indices to the global 4x4 indices.
        
        # Local U indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
        # Global U indices for a 2x2 block at (r, c) in 4x4:
        # r in {0,1}, c in {0,1} for the block position in the 4x4.
        # Actually, the 4x4 is split into four 2x2 blocks.
        # Block (0,0): i in 0,1; j in 0,1.
        # Block (0,1): i in 0,1; j in 2,3.
        # Block (1,0): i in 2,3; j in 0,1.
        # Block (1,1): i in 2,3; j in 2,3.
        
        # We are adding a 2x2x2 decomposition for a specific block multiplication.
        # The inputs are linear combinations of the 4 blocks.
        # This is handled by the Strassen combination logic.
        
        pass

    # Instead of complex offsets, we will just generate the 49 terms directly.
    # We define the 7 vectors for 2x2x2 as above.
    u_2x2 = [
        [1,0,1,0],
        [0,1,1,0],
        [1,0,0,0],
        [0,0,1,0],
        [1,1,0,0],
        [0,1,-1,0],
        [0,1,0,-1]
    ]
    v_2x2 = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [0,1,-1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    w_2x2 = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    
    # Strassen coefficients for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the block indices (0,1) to the 4x4 indices.
    # Block 0: i 0-1, j 0-1
    # Block 1: i 0-1, j 2-3
    # Block 2: i 2-3, j 0-1
    # Block 3: i 2-3, j 2-3
    
    # For each Strassen term, we add 7 rank-1 terms.
    # Each rank-1 term involves a specific A-block and B-block.
    
    # We will iterate through the 7 Strassen terms.
    # For each, we determine which A-blocks and B-blocks are involved and their coefficients.
    # Then for each of the 7 2x2x2 vectors, we add the scaled contribution.
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, scale, block_u, block_v, block_w):
        nonlocal col
        # block_u, block_v, block_w are the block indices (0-3) for A, B, C.
        # We need to map the local 2x2 indices to global 4x4 indices.
        
        # Global U index: i*4 + j.
        # Local 2x2 index: li, lj.
        # Block (br, bc) -> i = br*2 + li, j = bc*2 + lj.
        
        for li in range(2):
            for lj in range(2):
                idx_u = (block_u[0]*2 + li) * 4 + (block_u[1]*2 + lj)
                val_u = u_vec[li*2+lj] * scale
                if val_u != 0:
                    U[idx_u][col] += val_u
                    
        for lj in range(2):
            for lk in range(2):
                idx_v = (block_v[0]*2 + lj) * 4 + (block_v[1]*2 + lk)
                val_v = v_vec[lj*2+lk] * scale
                if val_v != 0:
                    V[idx_v][col] += val_v
                    
        for lk in range(2):
            for li in range(2):
                idx_w = (block_w[1]*2 + lk) * 4 + (block_w[0]*2 + li)
                # W index is k*n + i.
                # k is from B block, i is from A block.
                # Wait, W index in global is k*4 + i.
                # In 4x4x4, k is 0-3, i is 0-3.
                # Block (br, bc) for C?
                # C block (cr, cc) -> i = cr*2 + li, k = cc*2 + lk.
                # So idx_w = (cc*2 + lk) * 4 + (cr*2 + li).
                
                val_w = w_vec[lk*2+li] * scale
                if val_w != 0:
                    W[idx_w][col] += val_w
                    
        col += 1

    # Define the Strassen terms
    # (u_blocks, v_blocks, w_blocks, scale)
    # u_blocks: list of (block_idx, coeff)
    # v_blocks: list of (block_idx, coeff)
    # w_blocks: list of (block_idx, coeff)
    
    # This is getting complicated.
    # Simpler: Just use the schoolbook for 4x4x4? Rank 64.
    # Total 64 + 16 + 20 = 100.
    # We want 85.
    
    # Let's just implement the 49 terms explicitly.
    # I will hardcode the 49 terms for 4x4x4 Strassen.
    
    # For now, I will use the schoolbook for 4x4x4 to ensure correctness,
    # and then optimize if time permits.
    # But 100 is the seed.
    
    # Let's try to just do the 4x4x4 part with Strassen correctly.
    
    # I will write a function to generate the 49 terms.
    
    return {"U": U, "V": V, "W": W}
