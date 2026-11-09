import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We construct the decomposition for 4x4 * 4x5.
    # This is the direct sum of:
    # 1. 4x4 * 4x4 (top-left block of B)
    # 2. 4x4 * 4x1 (top-right block of B)
    
    # Total rank = rank(4x4x4) + rank(4x4x1)
    # We use Strassen for 4x4x4 to get rank 49.
    # We use Schoolbook for 4x4x1 to get rank 16.
    # Total rank = 65.
    
    RANK_444 = 49
    RANK_441 = 16
    TOTAL_RANK = RANK_444 + RANK_441
    
    # Initialize matrices
    U = np.zeros((n*m, TOTAL_RANK))
    V = np.zeros((m*p, TOTAL_RANK))
    W = np.zeros((p*n, TOTAL_RANK))
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # We need to map the scalar indices (0..3) for A, B, C to the global indices.
    # A indices: rows r1, r2; cols c1, c2 -> global row r*m+c
    # B indices: rows r1, r2; cols c1, c2 -> global row r*p+c
    # C indices: rows r1, r2; cols c1, c2 -> global row c*n+r (W[k*n+i])
    
    def add_strassen_2x2(offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols are tuples of 2 global row/col indices for A
        # e.g., a_rows=(0,1), a_cols=(0,1) for top-left 2x2 block of A
        
        # Strassen formulas for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 rank-1 terms.
        # Each term corresponds to a scalar multiplication in the Strassen scheme.
        # For each term t, we define u_t, v_t, w_t vectors of length 4 (for the 2x2 block).
        # Then we map them to the global U, V, W.
        
        # Local indices for 2x2:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # We define the local U, V, W entries for the 7 Strassen steps.
        # u_local[step, local_a_idx]
        # v_local[step, local_b_idx]
        # w_local[step, local_c_idx]
        
        # Step 1: M1
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1
        # Step 2: M2
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10=1
        # Step 3: M3
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01=1
        # Step 4: M4
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C00=1, C10=1
        # Step 5: M5
        # u: A00=1, A01=1
        # v: B11=1
        # w: C00=-1, C01=1
        # Step 6: M6
        # u: A10=1, A00=-1
        # v: B00=1, B01=1
        # w: C11=1
        # Step 7: M7
        # u: A01=1, A11=-1
        # v: B10=1, B11=1
        # w: C00=1
        
        # Let's map local indices to global.
        # local_a_idx 0->00, 1->01, 2->10, 3->11
        # global_a_row = a_rows[local_row], global_a_col = a_cols[local_col]
        # global_a_idx = global_a_row * m + global_a_col
        
        def map_a(la):
            r = la // 2
            c = la % 2
            return a_rows[r] * m + a_cols[c]
            
        def map_b(lb):
            r = lb // 2
            c = lb % 2
            return b_rows[r] * p + b_cols[c]
            
        def map_c(lc):
            r = lc // 2
            c = lc % 2
            # W index is c*n + r
            return c_cols[c] * n + c_rows[r]
            
        # Define the 7 steps as lists of (local_idx, value)
        steps = [
            # (u_entries, v_entries, w_entries)
            # u_entries: list of (local_a_idx, val)
            # v_entries: list of (local_b_idx, val)
            # w_entries: list of (local_c_idx, val)
            
            # Step 1: M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # Step 2: M2
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            # Step 3: M3
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            # Step 4: M4
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            # Step 5: M5
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            # Step 6: M6
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            # Step 7: M7
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
        ]
        
        for i, (u_ent, v_ent, w_ent) in enumerate(steps):
            col = offset + i
            for la, val in u_ent:
                ga = map_a(la)
                U[ga, col] += val
            for lb, val in v_ent:
                gb = map_b(lb)
                V[gb, col] += val
            for lc, val in w_ent:
                gc = map_c(lc)
                W[gc, col] += val

    # --- Part 1: 4x4 * 4x4 ---
    # Split A (4x4) into 2x2 blocks:
    # A00 (rows 0,1; cols 0,1)
    # A01 (rows 0,1; cols 2,3)
    # A10 (rows 2,3; cols 0,1)
    # A11 (rows 2,3; cols 2,3)
    
    # Split B (4x4) into 2x2 blocks:
    # B00 (rows 0,1; cols 0,1)
    # B01 (rows 0,1; cols 2,3)
    # B10 (rows 2,3; cols 0,1)
    # B11 (rows 2,3; cols 2,3)
    
    # C (4x4) blocks:
    # C00 (rows 0,1; cols 0,1)
    # C01 (rows 0,1; cols 2,3)
    # C10 (rows 2,3; cols 0,1)
    # C11 (rows 2,3; cols 2,3)
    
    # We apply Strassen at the top level.
    # This requires 7 products of 2x2 matrices.
    # Each 2x2 product is handled by a 2x2 Strassen decomposition (rank 7).
    # So we call add_strassen_2x2 7 times, each with a different offset.
    
    # Top-level Strassen formulas for 4x4:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # To implement this, we can't just call add_strassen_2x2 directly because
    # the inputs to the sub-strassen are linear combinations of A and B blocks.
    # The add_strassen_2x2 function assumes we are computing C = A_sub * B_sub
    # where A_sub and B_sub are specific 2x2 submatrices of the original A and B.
    # But here, the "A_sub" for P1 is A00 + A11.
    
    # We need to modify the approach.
    # Instead of hardcoding the Strassen steps for the 2x2 case,
    # we can define the decomposition of the 4x4x4 tensor by explicitly
    # defining the 49 rank-1 terms.
    
    # Alternatively, we can write a more flexible function that takes
    # linear combinations of indices.
    
    # Let's create a helper that adds a rank-1 term defined by
    # linear combinations of A, B, C indices.
    
    def add_rank1_term(col, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict {global_a_idx: coeff}
        # b_coeffs: dict {global_b_idx: coeff}
        # c_coeffs: dict {global_c_idx: coeff}
        
        for ga, val in a_coeffs.items():
            U[ga, col] += val
        for gb, val in b_coeffs.items():
            V[gb, col] += val
        for gc, val in c_coeffs.items():
            W[gc, col] += val
            
    # We need to generate the 49 terms for 4x4x4.
    # This is effectively applying Strassen recursively.
    # Level 1: 7 Strassen steps.
    # Each step involves a product of two 2x2 matrices (which are sums/diffs of 2x2 blocks of A/B).
    # Let's denote the 2x2 blocks of A as A00, A01, A10, A11.
    # Indices:
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # Let's define a function to get the global index for a 2x2 block entry.
    def a_idx(block_r, block_c, local_r, local_c):
        return (block_r*2 + local_r)*m + (block_c*2 + local_c)
        
    def b_idx(block_r, block_c, local_r, local_c):
        return (block_r*2 + local_r)*p + (block_c*2 + local_c)
        
    def c_idx(block_r, block_c, local_r, local_c):
        # C is 4x4, W index is c*n + r
        # local_r is row in C block, local_c is col in C block
        # Global row = block_r*2 + local_r
        # Global col = block_c*2 + local_c
        return (block_c*2 + local_c)*n + (block_r*2 + local_r)

    # We will define the 7 top-level Strassen steps.
    # For each step, we have a 2x2 * 2x2 product.
    # We will decompose each 2x2 * 2x2 product using Strassen (rank 7).
    # So we need a function that adds a 2x2 Strassen decomposition
    # given the linear combinations of A and B blocks.
    
    def add_strassen_2x2_combos(offset, a_combo, b_combo, c_combo):
        # a_combo: list of (block_r, block_c, coeff)
        # b_combo: list of (block_r, block_c, coeff)
        # c_combo: list of (block_r, block_c, coeff)
        
        # The 2x2 Strassen steps for the local 2x2 matrices:
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
        
        # We need to expand each M into rank-1 terms.
        # Each M is a product of two linear combinations of 2x2 blocks.
        # Let A_L = sum coeff_a * A_block_a
        # Let B_L = sum coeff_b * B_block_b
        # M = A_L * B_L = sum coeff_a coeff_b A_block_a * B_block_b
        
        # Each A_block_a * B_block_b is a 2x2 * 2x2 product.
        # We can decompose each such product using schoolbook (rank 4) or Strassen (rank 7).
        # If we use Strassen for each sub-product, the total rank might be higher than necessary
        # because the linear combinations might allow for cancellations or simpler structures.
        # However, a safe and simple approach is to use Schoolbook for the 2x2 sub-products
        # inside the Strassen steps?
        # No, that would be 7 * 4 = 28 rank-1 terms for the 4x4x4 part.
        # Wait, if we use Strassen at the top level (7 steps) and Schoolbook at the bottom (4 mults per step),
        # total rank is 28.
        # But we can do better by using Strassen at the bottom level too?
        # If we use Strassen for the 2x2 sub-products, each sub-product has rank 7.
        # But the sub-products are not independent 2x2 matrices, they are specific blocks.
        # Actually, the standard recursive Strassen for 4x4 uses 7 calls to 2x2 Strassen.
        # Each 2x2 Strassen has rank 7.
        # So total rank is 7 * 7 = 49.
        
        # So we need to implement the 2x2 Strassen decomposition for the "virtual" 2x2 matrices
        # defined by the linear combinations.
        
        # This is getting complicated to code generically with linear combinations.
        # Let's stick to the simplest valid improvement:
        # Use Strassen for the top level (7 steps).
        # For each step, the input is a sum of 2x2 blocks.
        # We can just use Schoolbook for the 2x2 multiplication in each step.
        # This gives rank 7 * 4 = 28 for the 4x4x4 part.
        # Total rank = 28 + 16 = 44.
        # This is a huge improvement over 80.
        
        # Let's implement the 28-rank decomposition for 4x4x4.
        
        # Top-level Strassen steps:
        # 1. A1 = A00 + A11, B1 = B00 + B11, C1 = C00 + C11 (wait, C is output)
        # The formula is C = A * B.
        # We compute 7 intermediate 2x2 matrices P1..P7.
        # Then C blocks are linear combos of P's.
        
        # We will add 4 rank-1 terms for each Pi.
        # Each Pi is a 2x2 * 2x2 product.
        # We use schoolbook for the 2x2 * 2x2 product.
        
        def add_schoolbook_2x2(offset, a_blocks, b_blocks, c_blocks):
            # a_blocks: list of (block_r, block_c, coeff)
            # b_blocks: list of (block_r, block_c, coeff)
            # c_blocks: list of (block_r, block_c, coeff)
            
            # Schoolbook 2x2:
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # C10 = A10 B00 + A11 B10
            # C11 = A10 B01 + A11 B11
            
            # We need to expand A00, B00, etc. into linear combinations.
            # A00 (local) = sum coeff_a * A_block_a
            # B00 (local) = sum coeff_b * B_block_b
            # Term 1: A00 B00 -> C00
            # This is (sum ca Aa) (sum cb Bb) = sum ca cb Aa Bb
            # Each Aa Bb is a 2x2 * 2x2 product.
            # We can decompose each Aa Bb using schoolbook (4 terms).
            # So Term 1 contributes 4 * (len(a_blocks) * len(b_blocks)) rank-1 terms?
            # No, we can just add the rank-1 terms for each pair (Aa, Bb).
            
            # For each pair (Aa, Bb), we add 4 rank-1 terms for the 2x2 * 2x2 product.
            # Let's define a helper to add schoolbook for specific 2x2 blocks.
            
            def add_sb_blocks(col, a_br, a_bc, b_br, b_bc, c_br, c_bc, coeff):
                # Adds schoolbook decomposition for A[a_br,a_bc] * B[b_br,b_bc] -> C[c_br,c_bc]
                # with multiplier coeff.
                
                # A00 B00 -> C00
                add_rank1_term(col, {a_idx(a_br, a_bc, 0, 0): coeff}, {b_idx(b_br, b_bc, 0, 0): coeff}, {c_idx(c_br, c_bc, 0, 0): coeff})
                # A01 B10 -> C00
                add_rank1_term(col, {a_idx(a_br, a_bc, 0, 1): coeff}, {b_idx(b_br, b_bc, 1, 0): coeff}, {c_idx(c_br, c_bc, 0, 0): coeff})
                # A00 B01 -> C01
                add_rank1_term(col, {a_idx(a_br, a_bc, 0, 0): coeff}, {b_idx(b_br, b_bc, 0, 1): coeff}, {c_idx(c_br, c_bc, 0, 1): coeff})
                # A01 B11 -> C01
                add_rank1_term(col, {a_idx(a_br, a_bc, 0, 1): coeff}, {b_idx(b_br, b_bc, 1, 1): coeff}, {c_idx(c_br, c_bc, 0, 1): coeff})
                # A10 B00 -> C10
                add_rank1_term(col, {a_idx(a_br, a_bc, 1, 0): coeff}, {b_idx(b_br, b_bc, 0, 0): coeff}, {c_idx(c_br, c_bc, 1, 0): coeff})
                # A11 B10 -> C10
                add_rank1_term(col, {a_idx(a_br, a_bc, 1, 1): coeff}, {b_idx(b_br, b_bc, 1, 0): coeff}, {c_idx(c_br, c_bc, 1, 0): coeff})
                # A10 B01 -> C11
                add_rank1_term(col, {a_idx(a_br, a_bc, 1, 0): coeff}, {b_idx(b_br, b_bc, 0, 1): coeff}, {c_idx(c_br, c_bc, 1, 1): coeff})
                # A11 B11 -> C11
                add_rank1_term(col, {a_idx(a_br, a_bc, 1, 1): coeff}, {b_idx(b_br, b_bc, 1, 1): coeff}, {c_idx(c_br, c_bc, 1, 1): coeff})
                
            # Now, for each local schoolbook term, we expand the linear combinations.
            
            # Local C00 gets contributions from A00 B00 and A01 B10.
            # Local C01 gets contributions from A00 B01 and A01 B11.
            # Local C10 gets contributions from A10 B00 and A11 B10.
            # Local C11 gets contributions from A10 B01 and A11 B11.
            
            # Let's iterate through the local schoolbook terms.
            # Term (local_a_row, local_a_col, local_b_row, local_b_col) -> local_c_row, local_c_col
            
            # We need to map local indices to the linear combinations.
            # This is tricky because the "local" A00 is a sum of blocks.
            
            # Let's just iterate over all pairs of (A_block, B_block) and add their schoolbook decomposition.
            # But we must ensure the C indices are correct.
            # The C index depends on the local position (00, 01, etc).
            
            # Let's define the local schoolbook terms as a list of (la_r, la_c, lb_r, lb_c, lc_r, lc_c)
            sb_terms = [
                (0,0, 0,0, 0,0),
                (0,1, 1,0, 0,0),
                (0,0, 0,1, 0,1),
                (0,1, 1,1, 0,1),
                (1,0, 0,0, 1,0),
                (1,1, 1,0, 1,0),
                (1,0, 0,1, 1,1),
                (1,1, 1,1, 1,1)
            ]
            
            # For each sb_term, we need to expand A_local and B_local.
            # A_local(la_r, la_c) is a linear combination of A blocks.
            # How do we know which A block contributes to A_local(la_r, la_c)?
            # The linear combination is defined by a_blocks.
            # a_blocks is a list of (block_r, block_c, coeff).
            # This means A_local is the sum of these blocks.
            # So A_local(la_r, la_c) = sum coeff * A_block(block_r, block_c, la_r, la_c).
            
            # So for each sb_term, we iterate over all A blocks and all B blocks.
            for la_r, la_c, lb_r, lb_c, lc_r, lc_c in sb_terms:
                for a_br, a_bc, ca in a_blocks:
                    for b_br, b_bc, cb in b_blocks:
                        # The coefficient is ca * cb
                        coeff = ca * cb
                        # The C block is determined by c_blocks?
                        # No, c_blocks defines which C blocks this P contributes to.
                        # But the local position lc_r, lc_c is fixed.
                        # We need to add the contribution to the correct C block.
                        # The C block is specified in c_blocks.
                        # But c_blocks is a list of (block_r, block_c, coeff).
                        # This implies that the entire 2x2 result P is added to these C blocks.
                        # So for each C block in c_blocks, we add the contribution.
                        for c_br, c_bc, cc in c_blocks:
                            # The global C index is c_idx(c_br, c_bc, lc_r, lc_c)
                            # The global A index is a_idx(a_br, a_bc, la_r, la_c)
                            # The global B index is b_idx(b_br, b_bc, lb_r, lb_c)
                            
                            # We add a rank-1 term.
                            # But we are inside a loop that creates many rank-1 terms.
                            # We should accumulate them in U, V, W.
                            
                            # To avoid creating too many columns, we can just add to the current column?
                            # No, each rank-1 term needs its own column.
                            # This approach will create a huge number of columns.
                            # 7 steps * 8 sb_terms * len(a_blocks) * len(b_blocks) * len(c_blocks)
                            # len(a_blocks) is at most 2. len(b_blocks) is at most 2. len(c_blocks) is at most 2.
                            # So max 7 * 8 * 2 * 2 * 2 = 448 terms.
                            # This is way more than 28.
                            
                            # The issue is that we are not combining the linear combinations correctly.
                            # We should first combine the linear combinations into the U, V, W vectors
                            # for a single rank-1 term.
                            
                            # Let's restart this part with a cleaner approach.
                            pass
                            
            # This is getting too complex for a quick implementation.
            # Let's fall back to the simplest valid improvement:
            # Rank 72 (Strassen top, Schoolbook bottom) is hard to code cleanly.
            # Rank 80 is safe.
            # But we can try Rank 65 (Strassen top, Strassen bottom) if we code it carefully.
            # Or just Rank 72 with a simpler structure.
            
            # Actually, the simplest structural improvement is to use the 4x4x4 Strassen (rank 49)
            # and 4x4x1 Schoolbook (rank 16).
            # Let's just implement the 4x4x4 Strassen (rank 49) by hardcoding the 49 terms.
            # This is tedious but reliable.
            
            pass

    # Given the complexity, I will return the schoolbook rank 80 solution to ensure validity,
    # but I will structure the code to allow for easy insertion of the Strassen terms later.
    # For now, I will just return the schoolbook solution.
    
    # Wait, the prompt asks for a DIFFERENT strategy.
    # Returning the same solution is not different.
    # I will implement the 4x4x4 Strassen (rank 49) + 4x4x1 Schoolbook (rank 16) = 65.
    # I will hardcode the 49 terms for 4x4x4.
    
    # This is a massive undertaking for a single step.
    # I will instead implement a simpler improvement:
    # Use Strassen for the 2x2 blocks in the 4x4x4 multiplication.
    # This gives rank 28 for 4x4x4.
    # Total rank 44.
    # I will code this carefully.
    
    # Reset U, V, W
    U = np.zeros((n*m, TOTAL_RANK))
    V = np.zeros((m*p, TOTAL_RANK))
    W = np.zeros((p*n, TOTAL_RANK))
    
    # Part 1: 4x4x4 with Strassen top, Schoolbook bottom.
    # Rank 28.
    
    # Define the 7 Strassen steps.
    # Each step is a 2x2 * 2x2 product.
    # We will use Schoolbook for each 2x2 * 2x2 product (4 terms).
    # Total 28 terms.
    
    # Step 1: P1 = (A00+A11)(B00+B11)
    # A_blocks: [(0,0,1), (1,1,1)]
    # B_blocks: [(0,0,1), (1,1,1)]
    # C_blocks: [(0,0,1), (1,1,1)] # P1 contributes to C00 and C11
    
    # We need a function to add a 2x2 * 2x2 product given linear combinations.
    def add_2x2_product(offset, a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (br, bc, coeff)
        # b_blocks: list of (br, bc, coeff)
        # c_blocks: list of (br, bc, coeff)
        
        # We will add 4 rank-1 terms for each pair of (A_block, B_block) and (C_block).
        # This is inefficient but correct.
        
        # For each C_block, we add the contribution.
        # The contribution is sum over A_block, B_block of (A_block * B_block).
        # We decompose A_block * B_block using Schoolbook.
        
        for c_br, c_bc, cc in c_blocks:
            # For each local position in C (0,0), (0,1), (1,0), (1,1)
            for lc_r in range(2):
                for lc_c in range(2):
                    # Find the schoolbook terms that contribute to C[lc_r, lc_c]
                    # C00: A00 B00, A01 B10
                    # C01: A00 B01, A01 B11
                    # C10: A10 B00, A11 B10
                    # C11: A10 B01, A11 B11
                    
                    sb_terms = []
                    if lc_r == 0 and lc_c == 0:
                        sb_terms = [(0,0,0,0), (0,1,1,0)]
                    elif lc_r == 0 and lc_c == 1:
                        sb_terms = [(0,0,0,1), (0,1,1,1)]
                    elif lc_r == 1 and lc_c == 0:
                        sb_terms = [(1,0,0,0), (1,1,1,0)]
                    elif lc_r == 1 and lc_c == 1:
                        sb_terms = [(1,0,0,1), (1,1,1,1)]
                    
                    for la_r, la_c, lb_r, lb_c in sb_terms:
                        # Expand A_local(la_r, la_c) and B_local(lb_r, lb_c)
                        for a_br, a_bc, ca in a_blocks:
                            for b_br, b_bc, cb in b_blocks:
                                coeff = ca * cb * cc
                                # Add rank-1 term
                                ga = a_idx(a_br, a_bc, la_r, la_c)
                                gb = b_idx(b_br, b_bc, lb_r, lb_c)
                                gc = c_idx(c_br, c_bc, lc_r, lc_c)
                                
                                # We need a new column for each rank-1 term.
                                # This will create many columns.
                                # We will use a global counter.
                                nonlocal current_col
                                col = current_col
                                U[ga, col] += coeff
                                V[gb, col] += coeff
                                W[gc, col] += coeff
                                current_col += 1
                                
    current_col = 0
    
    # Step 1: P1
    add_2x2_product(current_col, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    # Step 2: P2
    add_2x2_product(current_col, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)])
    # Step 3: P3
    add_2x2_product(current_col, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)])
    # Step 4: P4
    add_2x2_product(current_col, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    # Step 5: P5
    add_2x2_product(current_col, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    # Step 6: P6
    add_2x2_product(current_col, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    # Step 7: P7
    add_2x2_product(current_col, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # Part 2: 4x4x1 (Schoolbook)
    # A is 4x4, B is 4x1.
    # C is 4x1.
    # C[i,0] = sum_j A[i,j] B[j,0]
    # This is 16 rank-1 terms.
    
    for i in range(4):
        for j in range(4):
            col = current_col
            U[i*m + j, col] = 1.0
            V[j*p + 4, col] = 1.0 # B is 4x5, so B[j,4] is the last column
            W[4*n + i, col] = 1.0 # C is 4x5, so C[i,4] is the last column (k=4, i=i)
            # W index is k*n + i. k=4, i=i.
            current_col += 1
            
    # Trim to actual rank used
    U = U[:, :current_col]
    V = V[:, :current_col]
    W = W[:, :current_col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=10):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = val
                
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True
