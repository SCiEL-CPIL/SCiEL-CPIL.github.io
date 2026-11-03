import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    def get_schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen 2x2x2
    def get_strassen_2x2x2():
        # n=2, m=2, p=2
        # Terms:
        # M1 = A00(B00+B11)
        # M2 = (A00+A01)B11
        # M3 = (A10+A11)B00
        # M4 = A11(B10-B00)
        # M5 = (A00+A11)(B00+B11)
        # M6 = (A01-A11)(B10+B11)
        # M7 = (A00-A10)(B00+B01)
        
        # C00 = M5 - M4 + M7
        # C01 = M1 - M2 + M6
        # C10 = M3 - M1 + M4
        # C11 = M5 - M3 + M2
        
        # U indices: 00, 01, 10, 11
        # V indices: 00, 01, 10, 11
        # W indices: 00, 01, 10, 11 (k*2+i)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: A00(B00+B11)
        U[0, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[1, 0] = 1  # C01 += M1
        
        # M2: (A00+A01)B11
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1
        W[1, 1] = -1 # C01 -= M2
        
        # M3: (A10+A11)B00
        U[2, 2] = 1; U[3, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1  # C10 += M3
        
        # M4: A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1 # C00 -= M4
        W[2, 3] = 1  # C10 += M4
        
        # M5: (A00+A11)(B00+B11)
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = 1  # C00 += M5
        W[3, 4] = 1  # C11 += M5
        
        # M6: (A01-A11)(B10+B11)
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[1, 5] = 1  # C01 += M6
        
        # M7: (A00-A10)(B00+B01)
        U[0, 6] = 1; U[2, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1  # C00 += M7
        
        return U, V, W

    # Strassen 4x4x4
    def get_strassen_4x4x4():
        # Use 7 calls to 2x2x2 Strassen
        U_2, V_2, W_2 = get_strassen_2x2x2()
        R = 49
        U = np.zeros((16, R))
        V = np.zeros((16, R))
        W = np.zeros((16, R))
        
        # Block indices for 4x4:
        # A blocks: 00 (rows 0-1, cols 0-1), 01 (rows 0-1, cols 2-3), 10 (rows 2-3, cols 0-1), 11 (rows 2-3, cols 2-3)
        # B blocks: 00, 01, 10, 11
        # C blocks: 00, 01, 10, 11
        
        # For each of the 7 Strassen terms, we need to compute the product of two 2x2 blocks.
        # The term index t (0..6) determines which blocks of A and B are involved.
        # The internal 2x2x2 decomposition then determines the specific scalar multiplications.
        
        # Mapping of Strassen terms to block operations:
        # M1: A00 * (B00+B11) -> A_block 00, B_blocks 00, 11
        # M2: (A00+A01) * B11 -> A_blocks 00, 01, B_block 11
        # M3: (A10+A11) * B00 -> A_blocks 10, 11, B_block 00
        # M4: A11 * (B10-B00) -> A_block 11, B_blocks 10, 00
        # M5: (A00+A11) * (B00+B11) -> A_blocks 00, 11, B_blocks 00, 11
        # M6: (A01-A11) * (B10+B11) -> A_blocks 01, 11, B_blocks 10, 11
        # M7: (A00-A10) * (B00+B01) -> A_blocks 00, 10, B_blocks 00, 01
        
        # We iterate through the 7 terms.
        # For each term, we have a specific combination of A and B blocks.
        # We use the U_2, V_2, W_2 from the 2x2x2 case.
        # The rows of U_2 correspond to the elements of the 2x2 A block.
        # The rows of V_2 correspond to the elements of the 2x2 B block.
        # The rows of W_2 correspond to the elements of the 2x2 C block.
        
        # We need to map the local 2x2 indices to the global 4x4 indices.
        # Local A index (0..3) -> Global A index.
        # Local B index (0..3) -> Global B index.
        # Local C index (0..3) -> Global C index.
        
        # Term 0: M1 = A00(B00+B11)
        # A block 00: rows 0-1, cols 0-1.
        # B block 00: rows 0-1, cols 0-1. B block 11: rows 2-3, cols 2-3.
        # C block 01: rows 0-1, cols 2-3. (Wait, C01 is top-right)
        # In Strassen 2x2x2, W[1] corresponds to C01.
        
        # This manual mapping is error-prone. 
        # Alternative: Just use schoolbook for 4x4x4? Rank 64.
        # Total rank with schoolbook 4x4x4: 64 + 16 + 16 + 4 = 100. No improvement.
        
        # Let's try to implement the 4x4x4 Strassen carefully.
        
        # Helper to add a 2x2x2 decomposition to the global matrices
        def add_2x2x2(U_glob, V_glob, W_glob, offset, a_block, b_block, c_block, scale_a=1, scale_b=1, scale_c=1):
            # a_block: (row_start, col_start) for the 2x2 A block
            # b_block: (row_start, col_start) for the 2x2 B block
            # c_block: (row_start, col_start) for the 2x2 C block
            
            for r in range(7):
                col = offset + r
                # U
                for i_loc in range(4):
                    if U_2[i_loc, r] != 0:
                        i_glob_row = a_block[0] + (i_loc // 2)
                        i_glob_col = a_block[1] + (i_loc % 2)
                        U_glob[i_glob_row * 4 + i_glob_col, col] += U_2[i_loc, r] * scale_a
                # V
                for j_loc in range(4):
                    if V_2[j_loc, r] != 0:
                        j_glob_row = b_block[0] + (j_loc // 2)
                        j_glob_col = b_block[1] + (j_loc % 2)
                        V_glob[j_glob_row * 4 + j_glob_col, col] += V_2[j_loc, r] * scale_b
                # W
                for k_loc in range(4):
                    if W_2[k_loc, r] != 0:
                        k_glob_row = c_block[0] + (k_loc // 2)
                        k_glob_col = c_block[1] + (k_loc % 2)
                        # W index is k*n + i. Here n=4.
                        # Local W index k_loc is k_loc_row * 2 + k_loc_col?
                        # No, in 2x2x2, W index is k*2+i.
                        # k_loc is the index in the 4-element vector.
                        # k_loc = k*2 + i.
                        # So k = k_loc // 2, i = k_loc % 2.
                        k_glob = c_block[0] + (k_loc // 2)
                        i_glob = c_block[1] + (k_loc % 2)
                        W_glob[k_glob * 4 + i_glob, col] += W_2[k_loc, r] * scale_c

        # Define blocks
        # A blocks: (0,0), (0,2), (2,0), (2,2)
        # B blocks: (0,0), (0,2), (2,0), (2,2)
        # C blocks: (0,0), (0,2), (2,0), (2,2)
        
        # Term 0: M1 = A00(B00+B11)
        # A00: (0,0). B00: (0,0), B11: (2,2). C01: (0,2).
        # Wait, M1 contributes to C01.
        # In the 2x2x2 decomposition, the term M1 (index 0) has W[1]=1.
        # W[1] corresponds to C01 in the 2x2 output.
        # So we map the local C01 to the global C block.
        # The local C block is the one corresponding to the output of this term.
        # For M1, the output is C01.
        # So c_block should be (0,2).
        
        # However, the U and V blocks are fixed by the term.
        # M1 uses A00 and B00, B11.
        # But U_2 for term 0 is [1,0,0,0] (A00).
        # V_2 for term 0 is [1,0,0,1] (B00, B11).
        # This implies we need to add two contributions for V?
        # No, V_2 is a vector. It has non-zeros at 0 and 3.
        # 0 -> B00 (local). 3 -> B11 (local).
        # But B00 and B11 are different global blocks!
        # My `add_2x2x2` function assumes a single b_block.
        # This approach fails if the term involves multiple distinct global blocks for B or A.
        
        # I must handle each non-zero in U_2, V_2, W_2 individually.
        
        def add_term(U_glob, V_glob, W_glob, offset, term_idx, a_block, b_block, c_block):
            # term_idx: 0..6
            # a_block, b_block, c_block: (r, c) starts
            # This only works if the term uses a single A block, single B block, single C block.
            # Strassen terms do NOT always do this.
            # M1: A00 (single), B00+B11 (two), C01 (single).
            # So I need to decompose the term into sub-terms.
            
            # U_2[:, term_idx] has non-zeros at specific local indices.
            # V_2[:, term_idx] has non-zeros at specific local indices.
            # W_2[:, term_idx] has non-zeros at specific local indices.
            
            # For each non-zero in U, V, W, we map to global.
            # But the mapping depends on which global block the local index belongs to.
            # Since a term like M1 uses A00, all non-zeros in U_2[:, 0] map to A00.
            # But V_2[:, 0] has non-zeros at 0 (B00) and 3 (B11).
            # So I need to know which global block each local index maps to.
            
            # Let's pass a mapping function or dictionary.
            pass

        # This is getting complex. 
        # Given the 120s limit and the risk of bugs, 
        # maybe I should stick to a simpler improvement?
        
        # Is there a rank < 100 solution that is easier to implement?
        # What if I just use Strassen for 2x2x2 blocks in a 2x2 block decomposition of the 4x5x5?
        
        # Split n=4 into 2+2.
        # Split m=5 into 2+3.
        # Split p=5 into 2+3.
        
        # This leads to many sub-problems.
        
        # Let's try the 4+1 split again, but use a simpler 4x4x4 decomposition.
        # If I can't implement 4x4x4 Strassen correctly, I can use the "Winograd" or other known identities?
        
        # Actually, I can just use the schoolbook for 4x4x4 (Rank 64).
        # Total rank 100. No improvement.
        
        # What if I use Strassen for 2x2x2 parts only?
        # In the 4+1 split:
        # C_L = A_L B_LL + A_R B_RL
        # C_R = A_L B_LR + A_R B_RR
        
        # A_L is 4x4. B_LL is 4x4.
        # Can I decompose A_L B_LL using 2x2 Strassen blocks?
        # Yes, that's what 4x4x4 Strassen is.
        
        # Let's try to implement the 4x4x4 Strassen correctly by handling the block mappings explicitly.
        
        U_4 = np.zeros((16, 49))
        V_4 = np.zeros((16, 49))
        W_4 = np.zeros((16, 49))
        
        # Term 0: M1 = A00(B00+B11) -> C01
        # U: A00. V: B00, B11. W: C01.
        # U_2[:, 0] = [1, 0, 0, 0]. Non-zero at 0.
        # V_2[:, 0] = [1, 0, 0, 1]. Non-zero at 0, 3.
        # W_2[:, 0] = [0, 1, 0, 0]. Non-zero at 1.
        
        # A00 block: (0,0). Local index 0 -> Global (0,0).
        # B00 block: (0,0). Local index 0 -> Global (0,0).
        # B11 block: (2,2). Local index 3 -> Global (2+1, 2+1) = (3,3)?
        # Wait, local index 3 in 2x2 is (1,1).
        # So B11 local (1,1) -> Global (2+1, 2+1) = (3,3).
        # C01 block: (0,2). Local index 1 in 2x2 W (k=0, i=1) -> Global (0+0, 2+1) = (0,3)?
        # W index is k*n+i.
        # Local W index 1: k=0, i=1.
        # Global C block (0,2).
        # Global k = 0+0=0. Global i = 2+1=3.
        # Global W index = 0*4 + 3 = 3.
        
        # Let's write a helper to add a specific term with explicit block mappings.
        
        def add_strassen_term(U_g, V_g, W_g, col, term_idx, a_blocks, b_blocks, c_blocks):
            # a_blocks: dict {local_idx: (r, c)}
            # b_blocks: dict {local_idx: (r, c)}
            # c_blocks: dict {local_idx: (r, c)}
            
            for i_loc in range(4):
                val = U_2[i_loc, term_idx]
                if val != 0:
                    r, c = a_blocks[i_loc]
                    U_g[r * 4 + c, col] += val
                    
            for j_loc in range(4):
                val = V_2[j_loc, term_idx]
                if val != 0:
                    r, c = b_blocks[j_loc]
                    V_g[r * 4 + c, col] += val
                    
            for k_loc in range(4):
                val = W_2[k_loc, term_idx]
                if val != 0:
                    r, c = c_blocks[k_loc]
                    # W index is k*n + i.
                    # r is k, c is i.
                    W_g[r * 4 + c, col] += val

        # Term 0: M1
        # A00: local 0 -> (0,0).
        # B00: local 0 -> (0,0). B11: local 3 -> (3,3).
        # C01: local 1 -> (0,3).
        a0 = {0: (0,0)}
        b0 = {0: (0,0), 3: (3,3)}
        c0 = {1: (0,3)}
        add_strassen_term(U_4, V_4, W_4, 0, 0, a0, b0, c0)
        
        # Term 1: M2 = (A00+A01)B11 -> C01 (with -)
        # A00: local 0 -> (0,0). A01: local 1 -> (0,2).
        # B11: local 3 -> (3,3).
        # C01: local 1 -> (0,3).
        a1 = {0: (0,0), 1: (0,2)}
        b1 = {3: (3,3)}
        c1 = {1: (0,3)}
        # Note: M2 has coefficient -1 in C01 sum.
        # So we multiply W by -1? Or just let the sum handle it?
        # The term is M2. C01 = ... - M2.
        # So we should add -1 * W_2 value.
        # I will handle the sign in the c_blocks mapping or a scale factor.
        # Let's add a scale_w parameter.
        
        # Re-define add_strassen_term with scale
        def add_strassen_term_scaled(U_g, V_g, W_g, col, term_idx, a_blocks, b_blocks, c_blocks, scale_w=1):
            for i_loc in range(4):
                val = U_2[i_loc, term_idx]
                if val != 0:
                    r, c = a_blocks[i_loc]
                    U_g[r * 4 + c, col] += val
                    
            for j_loc in range(4):
                val = V_2[j_loc, term_idx]
                if val != 0:
                    r, c = b_blocks[j_loc]
                    V_g[r * 4 + c, col] += val
                    
            for k_loc in range(4):
                val = W_2[k_loc, term_idx] * scale_w
                if val != 0:
                    r, c = c_blocks[k_loc]
                    W_g[r * 4 + c, col] += val

        # Term 0 (M1): scale 1
        add_strassen_term_scaled(U_4, V_4, W_4, 0, 0, a0, b0, c0, 1)
        
        # Term 1 (M2): scale -1
        add_strassen_term_scaled(U_4, V_4, W_4, 1, 1, a1, b1, c1, -1)
        
        # Term 2 (M3): (A10+A11)B00 -> C10 (scale 1)
        a2 = {2: (2,0), 3: (2,2)}
        b2 = {0: (0,0)}
        c2 = {2: (2,0)} # C10 local 2 (k=1, i=0) -> Global (2,0)
        add_strassen_term_scaled(U_4, V_4, W_4, 2, 2, a2, b2, c2, 1)
        
        # Term 3 (M4): A11(B10-B00) -> C00 (-1), C10 (+1)
        # W_2[:, 3] has non-zeros at 0 (C00) and 2 (C10).
        # C00 local 0 -> Global (0,0). Scale -1.
        # C10 local 2 -> Global (2,0). Scale +1.
        # This requires different scales for different local W indices.
        # My function uses a single scale_w.
        # I need to handle this case by splitting or modifying the function.
        
        # Let's modify add_strassen_term_scaled to accept a dict for c_scales.
        
        def add_strassen_term_dict(U_g, V_g, W_g, col, term_idx, a_blocks, b_blocks, c_blocks, c_scales):
            for i_loc in range(4):
                val = U_2[i_loc, term_idx]
                if val != 0:
                    r, c = a_blocks[i_loc]
                    U_g[r * 4 + c, col] += val
                    
            for j_loc in range(4):
                val = V_2[j_loc, term_idx]
                if val != 0:
                    r, c = b_blocks[j_loc]
                    V_g[r * 4 + c, col] += val
                    
            for k_loc in range(4):
                val = W_2[k_loc, term_idx]
                if val != 0:
                    r, c = c_blocks[k_loc]
                    W_g[r * 4 + c, col] += val * c_scales[k_loc]

        # Term 3 (M4)
        a3 = {3: (2,2)}
        b3 = {2: (2,0), 0: (0,0)} # B10 local 2 -> (2,0), B00 local 0 -> (0,0)
        # V_2[:, 3] is [0,0,1,-1].
        # So local 2 has val 1, local 0 has val -1.
        # This is handled by V_2 values.
        c3 = {0: (0,0), 2: (2,0)}
        c3_scales = {0: -1, 2: 1}
        add_strassen_term_dict(U_4, V_4, W_4, 3, 3, a3, b3, c3, c3_scales)
        
        # Term 4 (M5): (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
        a4 = {0: (0,0), 3: (2,2)}
        b4 = {0: (0,0), 3: (3,3)}
        c4 = {0: (0,0), 3: (2,2)} # C00 local 0, C11 local 3
        c4_scales = {0: 1, 3: 1}
        add_strassen_term_dict(U_4, V_4, W_4, 4, 4, a4, b4, c4, c4_scales)
        
        # Term 5 (M6): (A01-A11)(B10+B11) -> C01 (+1)
        a5 = {1: (0,2), 3: (2,2)}
        b5 = {2: (2,0), 3: (3,3)}
        c5 = {1: (0,3)}
        c5_scales = {1: 1}
        add_strassen_term_dict(U_4, V_4, W_4, 5, 5, a5, b5, c5, c5_scales)
        
        # Term 6 (M7): (A00-A10)(B00+B01) -> C00 (+1)
        a6 = {0: (0,0), 2: (2,0)}
        b6 = {0: (0,0), 1: (0,2)}
        c6 = {0: (0,0)}
        c6_scales = {0: 1}
        add_strassen_term_dict(U_4, V_4, W_4, 6, 6, a6, b6, c6, c6_scales)
        
        # Now we have U_4, V_4, W_4 for 4x4x4 with rank 49.
        
        # Combine with the rest of the 4x5x5 problem.
        # R_total = 49 + 16 + 16 + 4 = 85.
        
        U = np.zeros((20, 85))
        V = np.zeros((25, 85))
        W = np.zeros((20, 85))
        
        # Part 1: A_L B_LL (4x4x4)
        # U_4 (16x49) -> U (20x85)
        # U index i*5+j. i in 0..3, j in 0..3.
        # U_4 index i*4+j.
        for i in range(4):
            for j in range(4):
                U[i*5+j, 0:49] = U_4[i*4+j, :]
        
        # V_4 (16x49) -> V (25x85)
        # V index j*5+k. j in 0..3, k in 0..3.
        for j in range(4):
            for k in range(4):
                V[j*5+k, 0:49] = V_4[j*4+k, :]
        
        # W_4 (16x49) -> W (20x85)
        # W index k*4+i. k in 0..3, i in 0..3.
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W_4[k*4+i, :]
        
        # Part 2: A_R B_RL (4x1x4)
        # Rank 16.
        # i in 0..3, j=4, k in 0..3.
        # U index i*5+4.
        # V index 4*5+k = 20+k.
        # W index k*4+i.
        for i in range(4):
            for k in range(4):
                col = 49 + i*4 + k
                U[i*5+4, col] = 1.0
                V[20+k, col] = 1.0
                W[k*4+i, col] = 1.0
        
        # Part 3: A_L B_LR (4x4x1)
        # Rank 16.
        # i in 0..3, j in 0..3, k=4.
        # U index i*5+j.
        # V index j*5+4.
        # W index 4*4+i = 16+i.
        for i in range(4):
            for j in range(4):
                col = 65 + i*4 + j
                U[i*5+j, col] = 1.0
                V[j*5+4, col] = 1.0
                W[16+i, col] = 1.0
        
        # Part 4: A_R B_RR (4x1x1)
        # Rank 4.
        # i in 0..3, j=4, k=4.
        # U index i*5+4.
        # V index 4*5+4 = 24.
        # W index 4*4+i = 16+i.
        for i in range(4):
            col = 81 + i
            U[i*5+4, col] = 1.0
            V[24, col] = 1.0
            W[16+i, col] = 1.0
            
        # Verify
        # Tensor check
        tensor_vals = np.zeros(4*5*5)
        for r in range(85):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        tensor_vals[i*25 + j*5 + k] += u_r[i*5+j] * v_r[j*5+k] * w_r[k*4+i]
                        
        # Target is 1 for all i,j,k?
        # No, the tensor for matrix multiplication is 1 if the indices match the summation, 0 otherwise.
        # T[i,j,k] = 1 if we are summing A[i,j] B[j,k].
        # So T[i,j,k] should be 1 for all i,j,k.
        
        if not np.all(np.abs(tensor_vals - 1.0) < 1e-9):
            # Fallback
            U_f, V_f, W_f = get_schoolbook(4, 5, 5)
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
