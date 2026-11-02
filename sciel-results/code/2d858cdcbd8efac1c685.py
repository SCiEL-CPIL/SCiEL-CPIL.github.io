import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # We will build the decomposition for 4x4x4 using recursive Strassen (rank 49)
    # and combine with schoolbook for the remaining parts.
    
    # 1. Build 4x4x4 Strassen decomposition
    # I will hardcode the 49 terms for 4x4x4 to avoid complex recursion in the final code?
    # No, recursion is safer and shorter.
    
    def build_strassen(N):
        if N == 1:
            return [[1]], [[1]], [[1]]
        
        half = N // 2
        U_rec, V_rec, W_rec = build_strassen(half)
        
        # The 7 Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # We need to map the indices of the "half" decomposition to the "full" indices.
        # The "half" decomposition operates on indices 0..half-1.
        # We need to create 7 sets of terms, each scaled/shifted appropriately.
        
        rank_rec = len(U_rec[0])
        rank_full = 7 * rank_rec
        
        U = [[0] * rank_full for _ in range(N*N)]
        V = [[0] * rank_full for _ in range(N*N)]
        W = [[0] * rank_full for _ in range(N*N)]
        
        # Helper to add a term from the recursive decomposition
        # to the full decomposition with specific offsets and signs.
        def add_term(term_idx, U_offsets, V_offsets, W_offsets, signs):
            # U_offsets: list of (row_offset, col_offset) for each row in U_rec?
            # No, U_rec has shape (half*half, rank_rec).
            # We need to map each row of U_rec to a row in U.
            
            # This is getting complicated.
            # Let's use a different representation.
            # Store terms as a list of (u_row, v_row, w_row, coeff)
            
            pass
            
        # Simpler approach:
        # Generate the 49 terms explicitly for 4x4x4.
        # Since I can't easily write the recursion logic for the flat arrays 
        # without significant code, I will use a known property:
        # The rank of 4x4x4 is 17.
        # If I can't get 17, I can use the 49-term Strassen.
        
        # Let's try to implement the 49-term Strassen carefully.
        
        # For each of the 7 Strassen steps, we have a "sub-problem" of size half.
        # The sub-problem's U, V, W are applied to specific blocks of A, B, C.
        
        # Step 1: M1 = (A00+A11)(B00+B11)
        # A00 is block (0,0), A11 is block (1,1).
        # The "A" part of the term is A00 + A11.
        # In the tensor decomposition, this means for each term in the sub-decomposition:
        # U[i, t] contributes to A00[i] and A11[i]?
        # No, the sub-decomposition computes X*Y where X and Y are half-size matrices.
        # Here X = A00+A11 and Y = B00+B11.
        # So for each term t in the sub-decomp:
        #   U_full[row_A00, t] += U_rec[row_A00_in_sub, t_sub]
        #   U_full[row_A11, t] += U_rec[row_A11_in_sub, t_sub]
        #   V_full[row_B00, t] += V_rec[row_B00_in_sub, t_sub]
        #   V_full[row_B11, t] += V_rec[row_B11_in_sub, t_sub]
        #   W_full[row_C00, t] += W_rec[row_C00_in_sub, t_sub]
        #   W_full[row_C11, t] += W_rec[row_C11_in_sub, t_sub]
        #   (And other C blocks get contributions from other M's)
        
        # This is doable.
        
        # Let's define the 7 steps.
        # Each step has:
        # - A_blocks: list of (block_row, block_col, sign)
        # - B_blocks: list of (block_row, block_col, sign)
        # - C_blocks: list of (block_row, block_col, sign)
        
        steps = [
            # M1
            {
                'A': [(0,0,1), (1,1,1)],
                'B': [(0,0,1), (1,1,1)],
                'C': [(0,0,1), (1,1,1)] # Wait, C00 and C11 get M1?
                # C00 = M1 + M4 - M5 + M7
                # C11 = M1 - M2 + M3 + M6
                # So M1 contributes to C00 and C11.
            },
            # M2
            {
                'A': [(1,0,1), (1,1,1)],
                'B': [(0,0,1)],
                'C': [(1,0,1), (1,1,-1)] # C10 += M2, C11 -= M2
            },
            # M3
            {
                'A': [(0,0,1)],
                'B': [(1,0,1), (1,1,-1)],
                'C': [(0,1,1), (1,1,1)] # C01 += M3, C11 += M3
            },
            # M4
            {
                'A': [(1,1,1)],
                'B': [(0,1,1), (0,0,-1)],
                'C': [(0,0,1), (1,0,1)] # C00 += M4, C10 += M4
            },
            # M5
            {
                'A': [(0,0,1), (0,1,1)],
                'B': [(1,1,1)],
                'C': [(0,0,-1), (0,1,1)] # C00 -= M5, C01 += M5
            },
            # M6
            {
                'A': [(1,0,1), (0,0,-1)],
                'B': [(0,0,1), (1,0,1)],
                'C': [(1,1,1)] # C11 += M6
            },
            # M7
            {
                'A': [(0,1,1), (1,1,-1)],
                'B': [(1,1,1), (0,1,1)],
                'C': [(0,0,1)] # C00 += M7
            }
        ]
        
        # Now, for each step, we iterate over the sub-terms.
        # sub-term t_sub in U_rec, V_rec, W_rec.
        # U_rec has rows 0..half*half-1.
        # These correspond to local indices in the half-size matrix.
        # We need to map local index to global index.
        
        # Local index l -> (li, lj) where li = l // half, lj = l % half.
        # Global index for block (br, bc) is (br*half + li, bc*half + lj).
        
        for step_idx, step in enumerate(steps):
            base_rank = step_idx * rank_rec
            for t_sub in range(rank_rec):
                t_full = base_rank + t_sub
                
                # Process A
                for br, bc, sign in step['A']:
                    # For each row in U_rec
                    for r_sub in range(half*half):
                        val = U_rec[r_sub][t_sub]
                        if val == 0: continue
                        li = r_sub // half
                        lj = r_sub % half
                        r_full = (br * half + li) * N + (bc * half + lj)
                        U[r_full][t_full] += sign * val
                        
                # Process B
                for br, bc, sign in step['B']:
                    for r_sub in range(half*half):
                        val = V_rec[r_sub][t_sub]
                        if val == 0: continue
                        li = r_sub // half
                        lj = r_sub % half
                        r_full = (br * half + li) * N + (bc * half + lj)
                        V[r_full][t_full] += sign * val
                        
                # Process C
                for br, bc, sign in step['C']:
                    for r_sub in range(half*half):
                        val = W_rec[r_sub][t_sub]
                        if val == 0: continue
                        li = r_sub // half
                        lj = r_sub % half
                        r_full = (br * half + li) * N + (bc * half + lj)
                        W[r_full][t_full] += sign * val
                        
        return U, V, W

    U_444, V_444, W_444 = build_strassen(4)
    
    # Now we have a 4x4x4 decomposition.
    # We need to extend it to 4x5x5.
    
    # The 4x4x4 decomposition covers i=0..3, j=0..3, k=0..3.
    # We need to handle:
    # 1. i=0..3, j=0..3, k=4
    # 2. i=0..3, j=4, k=0..3
    # 3. i=0..3, j=4, k=4
    
    # We will append schoolbook terms for these.
    
    # Total rank so far: 49
    rank_current = len(U_444[0])
    
    # Extend U, V, W to full size 4x5=20, 5x5=25, 5x4=20
    # Wait, dimensions are n=4, m=5, p=5.
    # U is (n*m, rank) = (20, rank)
    # V is (m*p, rank) = (25, rank)
    # W is (p*n, rank) = (20, rank)
    
    # The 4x4x4 part uses:
    # U rows: i in 0..3, j in 0..3. (16 rows)
    # V rows: j in 0..3, k in 0..3. (16 rows)
    # W rows: k in 0..3, i in 0..3. (16 rows)
    
    # We need to pad U, V, W to full size and add new columns for the remaining parts.
    
    # Let's create the final arrays.
    # We will add 16 + 16 + 4 = 36 more terms.
    # Total rank = 49 + 36 = 85.
    
    U_final = [[0] * 85 for _ in range(20)]
    V_final = [[0] * 85 for _ in range(25)]
    W_final = [[0] * 85 for _ in range(20)]
    
    # Copy 4x4x4 part
    for r in range(16):
        for c in range(49):
            U_final[r][c] = U_444[r][c]
            V_final[r][c] = V_444[r][c]
            W_final[r][c] = W_444[r][c]
            
    # Part 1: A_{4x4} B_{4x1}
    # i in 0..3, j in 0..3, k=4
    # Term for each i, j:
    # U[i*5+j, col] = 1
    # V[j*5+4, col] = 1
    # W[4*4+i, col] = 1
    
    col = 49
    for i in range(4):
        for j in range(4):
            U_final[i*5+j][col] = 1
            V_final[j*5+4][col] = 1
            W_final[4*4+i][col] = 1
            col += 1
            
    # Part 2: A_{4x1} B_{1x4}
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U_final[i*5+4][col] = 1
            V_final[4*5+k][col] = 1
            W_final[k*4+i][col] = 1
            col += 1
            
    # Part 3: A_{4x1} B_{1x1}
    # i in 0..3, j=4, k=4
    for i in range(4):
        U_final[i*5+4][col] = 1
        V_final[4*5+4][col] = 1
        W_final[4*4+i][col] = 1
        col += 1
        
    return {"U": U_final, "V": V_final, "W": W_final}
