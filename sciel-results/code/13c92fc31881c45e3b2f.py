import time
import numpy as np

DIMS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMS # 4, 4, 7

    # Helper to create a factor list
    def add_factor(U_list, V_list, W_list, u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # 1. Rank 7 decomposition for 2x2x2
    # Returns list of (u, v, w) where u,v,w are length 4
    # Indices: A: 0,1,2,3. B: 0,1,2,3. C: 0,1,2,3 (k*2+i)
    def get_2x2x2_rank7():
        factors = []
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        # A: [1,0,0,1], B: [1,0,0,1]
        # C: C00 (idx 0) +1, C11 (idx 3) +1
        factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2 = (A10+A11)B00 -> C10 + C11?
        # Standard Strassen:
        # M2 = (A21+A22)B11. A21=idx2, A22=idx3. B11=idx0.
        # C21 (idx 1) + C22 (idx 3)?
        # C21 = A21B11 + ...
        # Let's stick to the explicit M1..M7 definitions mapped to indices.
        
        # Re-doing carefully:
        # A indices: 00->0, 01->1, 10->2, 11->3
        # B indices: 00->0, 01->1, 10->2, 11->3
        # C indices: 00->0, 10->1, 01->2, 11->3 (k*2+i)
        
        # M1 = (A00+A11)(B00+B11)
        # A: 1,0,0,1. B: 1,0,0,1.
        # C00 += M1. C11 += M1.
        # W: 1,0,0,1
        factors[0] = ([1,0,0,1], [1,0,0,1], [1,0,0,1])
        
        # M2 = (A10+A11)B00
        # A: 0,0,1,1. B: 1,0,0,0.
        # C10 += M2. C11 += M2.
        # W: 0,1,0,1
        factors.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
        
        # M3 = A00(B10-B00)
        # A: 1,0,0,0. B: 0,0,1,-1.
        # C01 += M3. C11 += M3.
        # W: 0,0,1,1
        factors.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
        
        # M4 = A11(B11-B01)
        # A: 0,0,0,1. B: 0,-1,0,1.
        # C01 += M4. C11 += M4.
        # W: 0,0,1,1
        factors.append(([0,0,0,1], [0,-1,0,1], [0,0,1,1]))
        
        # M5 = (A00+A01)B11
        # A: 1,1,0,0. B: 0,0,0,1.
        # C00 += M5. C01 += M5.
        # W: 1,0,1,0
        factors.append(([1,1,0,0], [0,0,0,1], [1,0,1,0]))
        
        # M6 = (A10-A00)(B00+B01)
        # A: -1,0,1,0. B: 1,1,0,0.
        # C10 += M6. C11 += M6.
        # W: 0,1,0,1
        factors.append(([-1,0,1,0], [1,1,0,0], [0,1,0,1]))
        
        # M7 = (A01-A11)(B10+B11)
        # A: 0,1,0,-1. B: 0,0,1,1.
        # C00 += M7. C01 += M7.
        # W: 1,0,1,0
        factors.append(([0,1,0,-1], [0,0,1,1], [1,0,1,0]))
        
        return factors

    # 2. Rank 4 decomposition for 2x2x1
    # A: 2x2 (4), B: 2x1 (2), C: 1x2 (2 indices: k=0, i=0,1 -> 0,1)
    def get_2x2x1_rank4():
        factors = []
        # C00 = A00B0 + A01B1
        # M1: A00B0 -> C00
        factors.append(([1,0,0,0], [1,0], [1,0]))
        # M2: A01B1 -> C00
        factors.append(([0,1,0,0], [0,1], [1,0]))
        # M3: A10B0 -> C10
        factors.append(([0,0,1,0], [1,0], [0,1]))
        # M4: A11B1 -> C10
        factors.append(([0,0,0,1], [0,1], [0,1]))
        return factors

    # Main construction
    U_all = []
    V_all = []
    W_all = []

    # Part 1: 4x4x4 (Rank 49)
    # Split into 2x2 blocks.
    # Strassen steps on blocks.
    # Each step is a 2x2x2 mult.
    # We need to map local 2x2 indices to global 4x4 indices.
    
    # Global A (4x4):
    # Block (0,0) [0,1]x[0,1] -> Local A indices 0,1,2,3
    # Block (0,1) [0,1]x[2,3] -> Local A indices 0,1,2,3 (shifted)
    # Block (1,0) [2,3]x[0,1]
    # Block (1,1) [2,3]x[2,3]
    
    def map_2x2_to_4x4_A(local_vec, block_r, block_c):
        # local_vec is length 4: [00, 01, 10, 11]
        # block_r, block_c in {0,1}
        global_vec = [0]*16
        r0 = block_r * 2
        c0 = block_c * 2
        # 00 -> r0, c0
        global_vec[r0*4 + c0] = local_vec[0]
        # 01 -> r0, c0+1
        global_vec[r0*4 + c0 + 1] = local_vec[1]
        # 10 -> r0+1, c0
        global_vec[(r0+1)*4 + c0] = local_vec[2]
        # 11 -> r0+1, c0+1
        global_vec[(r0+1)*4 + c0 + 1] = local_vec[3]
        return global_vec

    def map_2x2_to_4x4_B(local_vec, block_r, block_c):
        # B is 4x4 in this part.
        global_vec = [0]*16
        r0 = block_r * 2
        c0 = block_c * 2
        global_vec[r0*4 + c0] = local_vec[0]
        global_vec[r0*4 + c0 + 1] = local_vec[1]
        global_vec[(r0+1)*4 + c0] = local_vec[2]
        global_vec[(r0+1)*4 + c0 + 1] = local_vec[3]
        return global_vec

    def map_2x2_to_4x4_C(local_vec, block_r, block_c):
        # C is 4x4. Index k*4+i.
        # Local C index k*2+i.
        global_vec = [0]*16
        r0 = block_r * 2
        c0 = block_c * 2
        # Local 0 (k=0,i=0) -> Global k=c0, i=r0?
        # Wait, C index is k*n+i.
        # Local C is 2x2. k_loc, i_loc.
        # Global k = c0 + k_loc. Global i = r0 + i_loc.
        # Global index = (c0 + k_loc)*4 + (r0 + i_loc).
        
        # local_vec[0] is k=0,i=0.
        global_vec[(c0+0)*4 + (r0+0)] = local_vec[0]
        # local_vec[1] is k=0,i=1.
        global_vec[(c0+0)*4 + (r0+1)] = local_vec[1]
        # local_vec[2] is k=1,i=0.
        global_vec[(c0+1)*4 + (r0+0)] = local_vec[2]
        # local_vec[3] is k=1,i=1.
        global_vec[(c0+1)*4 + (r0+1)] = local_vec[3]
        return global_vec

    # Strassen Block Indices
    # M1: (A00+A11)(B00+B11)
    # A00 is block(0,0). A11 is block(1,1).
    # B00 is block(0,0). B11 is block(1,1).
    # C00 is block(0,0). C11 is block(1,1).
    
    # We will iterate through the 7 Strassen steps.
    # For each step, we have a set of (A_block, B_block, C_block) contributions.
    # Actually, Strassen defines specific linear combos of blocks.
    
    # Let's define the 7 steps explicitly with block coordinates.
    # Step 1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+C(1,1)
    # Step 2: A(1,0)+A(1,1) * B(0,0) -> C(1,0)+C(1,1)
    # Step 3: A(0,0) * B(1,0)-B(0,0) -> C(0,1)+C(1,1)
    # Step 4: A(1,1) * B(1,1)-B(0,1) -> C(0,1)+C(1,1)
    # Step 5: A(0,0)+A(0,1) * B(1,1) -> C(0,0)+C(0,1)
    # Step 6: A(1,0)-A(0,0) * B(0,0)+B(0,1) -> C(1,0)+C(1,1)
    # Step 7: A(0,1)-A(1,1) * B(1,0)+B(1,1) -> C(0,0)+C(0,1)
    
    # This is getting complex to map to the 2x2x2 rank7 factors directly.
    # The 2x2x2 rank7 factors operate on the *elements* of the 2x2 blocks.
    # So for Step 1, we need to compute (A00+A11) * (B00+B11).
    # This is a product of two 2x2 matrices.
    # We can form the matrix A_sum = A00+A11 and B_sum = B00+B11.
    # Then use the 2x2x2 rank7 decomposition on A_sum and B_sum.
    # The result is C_sum, which contributes to C00 and C11.
    
    # So, for each Strassen step, we:
    # 1. Form the global U, V vectors for the 2x2x2 multiplication.
    # 2. Use the 7 factors from get_2x2x2_rank7().
    # 3. Map the local 2x2 indices to the global 4x4 indices based on which blocks are involved.
    
    # However, the "blocks" in Strassen are 2x2 matrices.
    # The 2x2x2 rank7 decomposition assumes the input is a single 2x2 matrix.
    # So if we want to compute (A00+A11)*(B00+B11), we treat (A00+A11) as the "A" matrix of the 2x2x2 problem.
    # The elements of A are $A_{00}[0,0] + A_{11}[0,0]$, etc.
    
    # So, for Step 1:
    # U_local for 2x2x2 will have entries like 1 for A00, 0 for A11, etc.
    # We need to map this to Global U.
    # Global U index for A00[0,0] is 0.
    # Global U index for A11[0,0] is 12 (row 2, col 2).
    
    # Let's write a helper to add a 2x2x2 factor to the global lists.
    
    def add_2x2x2_factor(global_U, global_V, global_W, 
                         u_local, v_local, w_local,
                         a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (block_r, block_c, sign)
        # b_blocks: list of (block_r, block_c, sign)
        # c_blocks: list of (block_r, block_c, sign)
        
        # u_local is length 4.
        # We need to sum the contributions.
        # But u_local is a vector for a *single* 2x2 matrix.
        # Here, the "A" matrix is a sum of blocks.
        # So U_global = sum( sign * map(u_local, block) )
        
        U_g = [0]*16
        for br, bc, sign in a_blocks:
            mapped = map_2x2_to_4x4_A(u_local, br, bc)
            for i in range(16):
                U_g[i] += sign * mapped[i]
                
        V_g = [0]*16
        for br, bc, sign in b_blocks:
            mapped = map_2x2_to_4x4_B(v_local, br, bc)
            for i in range(16):
                V_g[i] += sign * mapped[i]
                
        W_g = [0]*16
        for br, bc, sign in c_blocks:
            mapped = map_2x2_to_4x4_C(w_local, br, bc)
            for i in range(16):
                W_g[i] += sign * mapped[i]
                
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)

    # Define Strassen Steps
    # Step 1: (A00+A11)(B00+B11) -> C00+C11
    # A blocks: (0,0,1), (1,1,1)
    # B blocks: (0,0,1), (1,1,1)
    # C blocks: (0,0,1), (1,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
                         
    # Step 2: (A10+A11)B00 -> C10+C11
    # A blocks: (1,0,1), (1,1,1)
    # B blocks: (0,0,1)
    # C blocks: (1,0,1), (1,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)])
                         
    # Step 3: A00(B10-B00) -> C01+C11
    # A blocks: (0,0,1)
    # B blocks: (1,0,1), (0,0,-1)
    # C blocks: (0,1,1), (1,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(0,0,1)], [(1,0,1), (0,0,-1)], [(0,1,1), (1,1,1)])
                         
    # Step 4: A11(B11-B01) -> C01+C11
    # A blocks: (1,1,1)
    # B blocks: (1,1,1), (0,1,-1)
    # C blocks: (0,1,1), (1,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(1,1,1)], [(1,1,1), (0,1,-1)], [(0,1,1), (1,1,1)])
                         
    # Step 5: (A00+A01)B11 -> C00+C01
    # A blocks: (0,0,1), (0,1,1)
    # B blocks: (1,1,1)
    # C blocks: (0,0,1), (0,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)])
                         
    # Step 6: (A10-A00)(B00+B01) -> C10+C11
    # A blocks: (1,0,1), (0,0,-1)
    # B blocks: (0,0,1), (0,1,1)
    # C blocks: (1,0,1), (1,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1), (1,1,1)])
                         
    # Step 7: (A01-A11)(B10+B11) -> C00+C01
    # A blocks: (0,1,1), (1,1,-1)
    # B blocks: (1,0,1), (1,1,1)
    # C blocks: (0,0,1), (0,1,1)
    for u, v, w in get_2x2x2_rank7():
        add_2x2x2_factor(U_all, V_all, W_all, u, v, w,
                         [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)])

    # Part 2: 4x4x2 (Rank 28)
    # B is 4x2.
    # Split B into B_T (rows 0,1) and B_B (rows 2,3).
    # C_T = A11 B_T + A12 B_B
    # C_B = A21 B_T + A22 B_B
    
    # We need to map 2x2 B to 4x2 B.
    def map_2x2_to_4x2_B(local_vec, block_r):
        # block_r 0 or 1.
        # local_vec is 4 elements.
        # B is 4x2. Index j*2+k.
        # block_r=0: rows 0,1.
        # local 0 (0,0) -> global (0,0) idx 0
        # local 1 (0,1) -> global (0,1) idx 1
        # local 2 (1,0) -> global (1,0) idx 2
        # local 3 (1,1) -> global (1,1) idx 3
        global_vec = [0]*8
        r0 = block_r * 2
        global_vec[r0*2 + 0] = local_vec[0]
        global_vec[r0*2 + 1] = local_vec[1]
        global_vec[(r0+1)*2 + 0] = local_vec[2]
        global_vec[(r0+1)*2 + 1] = local_vec[3]
        return global_vec

    def map_2x2_to_4x2_C(local_vec, block_r):
        # C is 4x2. Index k*4+i.
        # local_vec is 4 elements (2x2).
        # block_r 0 or 1.
        global_vec = [0]*8
        r0 = block_r * 2
        # local 0 (k=0,i=0) -> global (k=0, i=r0) idx 0*4+r0
        global_vec[0*4 + r0] = local_vec[0]
        # local 1 (k=0,i=1) -> global (k=0, i=r0+1) idx 0*4+r0+1
        global_vec[0*4 + r0 + 1] = local_vec[1]
        # local 2 (k=1,i=0) -> global (k=1, i=r0) idx 1*4+r0
        global_vec[1*4 + r0] = local_vec[2]
        # local 3 (k=1,i=1) -> global (k=1, i=r0+1) idx 1*4+r0+1
        global_vec[1*4 + r0 + 1] = local_vec[3]
        return global_vec

    # C_T = A11 B_T + A12 B_B
    # Term 1: A11 B_T
    # A block (0,0). B block 0. C block 0.
    for u, v, w in get_2x2x2_rank7():
        U_g = map_2x2_to_4x4_A(u, 0, 0)
        V_g = map_2x2_to_4x2_B(v, 0)
        W_g = map_2x2_to_4x2_C(w, 0)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    # Term 2: A12 B_B
    # A block (0,1). B block 1. C block 0.
    for u, v, w in get_2x2x2_rank7():
        U_g = map_2x2_to_4x4_A(u, 0, 1)
        V_g = map_2x2_to_4x2_B(v, 1)
        W_g = map_2x2_to_4x2_C(w, 0)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    # C_B = A21 B_T + A22 B_B
    # Term 3: A21 B_T
    # A block (1,0). B block 0. C block 1.
    for u, v, w in get_2x2x2_rank7():
        U_g = map_2x2_to_4x4_A(u, 1, 0)
        V_g = map_2x2_to_4x2_B(v, 0)
        W_g = map_2x2_to_4x2_C(w, 1)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    # Term 4: A22 B_B
    # A block (1,1). B block 1. C block 1.
    for u, v, w in get_2x2x2_rank7():
        U_g = map_2x2_to_4x4_A(u, 1, 1)
        V_g = map_2x2_to_4x2_B(v, 1)
        W_g = map_2x2_to_4x2_C(w, 1)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)

    # Part 3: 4x4x1 (Rank 16)
    # B is 4x1.
    # Split B into v_T (rows 0,1) and v_B (rows 2,3).
    # C_T = A11 v_T + A12 v_B
    # C_B = A21 v_T + A22 v_B
    
    def map_2x2_to_4x1_B(local_vec, block_r):
        # local_vec length 2.
        global_vec = [0]*4
        r0 = block_r * 2
        global_vec[r0] = local_vec[0]
        global_vec[r0+1] = local_vec[1]
        return global_vec

    def map_2x2_to_4x1_C(local_vec, block_r):
        # local_vec length 2 (k=0, i=0,1).
        global_vec = [0]*4
        r0 = block_r * 2
        # local 0 (i=0) -> global i=r0
        global_vec[r0] = local_vec[0]
        # local 1 (i=1) -> global i=r0+1
        global_vec[r0+1] = local_vec[1]
        return global_vec

    # C_T = A11 v_T + A12 v_B
    for u, v, w in get_2x2x1_rank4():
        U_g = map_2x2_to_4x4_A(u, 0, 0)
        V_g = map_2x2_to_4x1_B(v, 0)
        W_g = map_2x2_to_4x1_C(w, 0)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    for u, v, w in get_2x2x1_rank4():
        U_g = map_2x2_to_4x4_A(u, 0, 1)
        V_g = map_2x2_to_4x1_B(v, 1)
        W_g = map_2x2_to_4x1_C(w, 0)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    # C_B = A21 v_T + A22 v_B
    for u, v, w in get_2x2x1_rank4():
        U_g = map_2x2_to_4x4_A(u, 1, 0)
        V_g = map_2x2_to_4x1_B(v, 0)
        W_g = map_2x2_to_4x1_C(w, 1)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)
        
    for u, v, w in get_2x2x1_rank4():
        U_g = map_2x2_to_4x4_A(u, 1, 1)
        V_g = map_2x2_to_4x1_B(v, 1)
        W_g = map_2x2_to_4x1_C(w, 1)
        add_factor(U_all, V_all, W_all, U_g, V_g, W_g)

    # Total Rank: 49 + 28 + 16 = 93.
    
    return {"U": U_all, "V": V_all, "W": W_all}

