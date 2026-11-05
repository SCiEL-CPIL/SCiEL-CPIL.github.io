import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2(off_i, off_j, off_k):
        # M1 = (A00+A11)(B00+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M2 = (A10+A11)B00
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # M3 = A00(B01-B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M4 = A11(B10-B00)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = -1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M5 = (A00+A01)B11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    # 4+1 partition: indices {0,1,2,3} for the 4x4 block, {4} for the 1x1
    # Use rank-49 Strassen decomposition for the 4x4x4 block
    # and schoolbook for the mixed blocks involving index 4

    # First, add the 49 rank-1 terms for the 4x4x4 block using recursive Strassen
    def add_strassen_4x4(off_i, off_j, off_k):
        # The 4x4x4 block uses indices off_i..off_i+3, off_j..off_j+3, off_k..off_k+3
        # We partition each dimension into two 2x2 blocks
        # This gives 7 "Strassen products" at the top level
        # Each top-level product is a product of two 2x2 matrices
        # Each 2x2 product uses 7 rank-1 terms (Strassen)
        # Total: 7 * 7 = 49 terms
        
        # Top-level Strassen combinations for 4x4:
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
        
        # The 7 top-level products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each top-level product, we need to compute the product of two 2x2 matrices
        # using Strassen's 7-term decomposition
        
        # Define the 7 top-level products with their coefficient patterns
        # Each product is of the form (sum of A blocks) * (sum of B blocks)
        # where the sums have coefficients +1 or -1
        
        top_level_products = [
            # (A_coeffs, B_coeffs) where A_coeffs[i,j] is the coefficient for A block (i,j)
            # and B_coeffs[i,j] is the coefficient for B block (i,j)
            # Block indices: 00, 01, 10, 11
            # M1 = (A00+A11)(B00+B11)
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}},
            # M2 = (A10+A11)B00
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}},
            # M3 = A00(B01-B11)
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}},
            # M4 = A11(B10-B00)
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}},
            # M5 = (A00+A01)B11
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}},
            # M6 = (A10-A00)(B00+B01)
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}},
            # M7 = (A01-A11)(B10+B11)
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}},
        ]
        
        # For each top-level product, we need to expand it into rank-1 terms
        # A top-level product is (sum_a c_a * A_block_a) * (sum_b d_b * B_block_b)
        # = sum_{a,b} c_a * d_b * A_block_a * B_block_b
        # Each A_block_a * B_block_b is a 2x2 matrix product
        # We use Strassen's 7-term decomposition for each 2x2 product
        
        # Strassen's 7 rank-1 terms for 2x2x2:
        # For matrices P (2x2) and Q (2x2), the 7 products are:
        # S1 = (P00+P11)(Q00+Q11)
        # S2 = (P10+P11)Q00
        # S3 = P00(Q01-Q11)
        # S4 = P11(Q10-Q00)
        # S5 = (P00+P01)Q11
        # S6 = (P10-P00)(Q00+Q01)
        # S7 = (P01-P11)(Q10+Q11)
        
        # For each pair (a, b) of blocks, we add 7 rank-1 terms
        # The coefficient is c_a * d_b
        
        strassen_2x2_terms = [
            # Each term is (P_coeffs, Q_coeffs, R_coeffs)
            # where P_coeffs[(i,j)] is the coefficient for P[i,j]
            # Q_coeffs[(i,j)] is the coefficient for Q[i,j]
            # R_coeffs[(i,j)] is the coefficient for R[i,j] in the output
            # S1
            {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'R': {(0,0): 1, (1,1): 1}},
            # S2
            {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'R': {(1,0): 1, (1,1): -1}},
            # S3
            {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'R': {(0,1): 1, (1,1): 1}},
            # S4
            {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'R': {(0,0): 1, (0,1): 1}},
            # S5
            {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'R': {(0,0): -1, (1,0): 1}},
            # S6
            {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'R': {(1,1): 1}},
            # S7
            {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'R': {(0,0): 1}},
        ]
        
        for prod in top_level_products:
            A_blocks = prod['A']
            B_blocks = prod['B']
            
            # Get the list of block indices
            A_idx = list(A_blocks.keys())
            B_idx = list(B_blocks.keys())
            
            for ai in A_idx:
                for bi in B_idx:
                    c_a = A_blocks[ai]
                    d_b = B_blocks[bi]
                    coeff = c_a * d_b
                    
                    # ai = (row_block, col_block) for A
                    # bi = (row_block, col_block) for B
                    # A_block_ai has rows off_i + 2*ai[0] to off_i + 2*ai[0]+1
                    # and cols off_j + 2*ai[1] to off_j + 2*ai[1]+1
                    # B_block_bi has rows off_j + 2*bi[0] to off_j + 2*bi[0]+1
                    # and cols off_k + 2*bi[1] to off_k + 2*bi[1]+1
                    # The product A_block_ai * B_block_bi has rows off_i + 2*ai[0] to off_i + 2*ai[0]+1
                    # and cols off_k + 2*bi[1] to off_k + 2*bi[1]+1
                    # Wait, that's not right.
                    # A is 4x4, partitioned into 2x2 blocks.
                    # A_block(ai[0], ai[1]) has rows 2*ai[0] to 2*ai[0]+1 (relative to off_i)
                    # and cols 2*ai[1] to 2*ai[1]+1 (relative to off_j)
                    # B_block(bi[0], bi[1]) has rows 2*bi[0] to 2*bi[0]+1 (relative to off_j)
                    # and cols 2*bi[1] to 2*bi[1]+1 (relative to off_k)
                    # For the product to be valid, we need ai[1] == bi[0]
                    # (the column index of A block must match the row index of B block)
                    # If ai[1] != bi[0], the product is zero (no contribution)
                    
                    if ai[1] != bi[0]:
                        continue
                    
                    # Now compute the 7 Strassen terms for this 2x2 product
                    # A_block has global row offset: off_i + 2*ai[0]
                    # A_block has global col offset: off_j + 2*ai[1]
                    # B_block has global row offset: off_j + 2*bi[0] = off_j + 2*ai[1]
                    # B_block has global col offset: off_k + 2*bi[1]
                    # Product has global row offset: off_i + 2*ai[0]
                    # Product has global col offset: off_k + 2*bi[1]
                    
                    a_row_off = off_i + 2*ai[0]
                    a_col_off = off_j + 2*ai[1]
                    b_row_off = off_j + 2*bi[0]
                    b_col_off = off_k + 2*bi[1]
                    prod_row_off = off_i + 2*ai[0]
                    prod_col_off = off_k + 2*bi[1]
                    
                    for sterm in strassen_2x2_terms:
                        P_coeffs = sterm['P']
                        Q_coeffs = sterm['Q']
                        R_coeffs = sterm['R']
                        
                        u = [0]*25
                        v = [0]*25
                        w = [0]*25
                        
                        # Fill u vector (for A matrix)
                        for (ri, ci), c in P_coeffs.items():
                            global_i = a_row_off + ri
                            global_j = a_col_off + ci
                            u[global_i * 5 + global_j] = coeff * c
                        
                        # Fill v vector (for B matrix)
                        for (ri, ci), c in Q_coeffs.items():
                            global_j = b_row_off + ri
                            global_k = b_col_off + ci
                            v[global_j * 5 + global_k] = coeff * c
                        
                        # Fill w vector (for C matrix)
                        for (ri, ci), c in R_coeffs.items():
                            global_k = prod_col_off + ci
                            global_i = prod_row_off + ri
                            w[global_k * 5 + global_i] = coeff * c
                        
                        add_col(u, v, w)
        
        # The 4x4x4 block uses indices 0-3 in each dimension
        # So off_i=0, off_j=0, off_k=0
        add_strassen_4x4(0, 0, 0)
        
        # Now add the mixed blocks involving index 4
        # These are:
        # - A[0:4,4] * B[4,0:4] -> C[0:4,0:4] : 4x1 * 1x4 = 4x4, rank 16
        # - A[0:4,4] * B[4,4] -> C[0:4,4] : 4x1 * 1x1 = 4x1, rank 4
        # - A[4,0:4] * B[0:4,0:4] -> C[4,0:4] : 1x4 * 4x4 = 1x4, rank 16
        # - A[4,0:4] * B[0:4,4] -> C[4,4] : 1x4 * 4x1 = 1x1, rank 4
        # - A[4,4] * B[4,0:4] -> C[4,0:4] : 1x1 * 4x4 = 4x4, rank 16
        # - A[4,4] * B[4,4] -> C[4,4] : 1x1 * 1x1 = 1x1, rank 1
        # - A[0:4,0:4] * B[4,0:4] -> C[0:4,0:4] : 4x4 * 4x4, but this is already covered
        # Wait, I need to be more careful.
        
        # The full 5x5x5 tensor has indices i,j,k in {0,1,2,3,4}
        # The 4x4x4 block covers i,j,k in {0,1,2,3}
        # The remaining terms involve at least one index equal to 4
        
        # Let's enumerate all (i,j,k) where at least one is 4:
        # Total 125 - 64 = 61 terms
        
        # For each such term, we use schoolbook (rank 1 per term)
        # But we can potentially group some of them
        
        # Actually, let's just use schoolbook for all remaining terms
        # This gives 61 additional terms, total 49 + 61 = 110
        
        # But wait, can we do better?
        # The mixed blocks are:
        # 1. i in {0..3}, j=4, k in {0..3}: A[i,4]*B[4,k] -> C[k,i]
        #    This is a 4x1 * 1x4 = 4x4 block, rank 16
        # 2. i in {0..3}, j=4, k=4: A[i,4]*B[4,4] -> C[4,i]
        #    This is a 4x1 * 1x1 = 4x1 block, rank 4
        # 3. i=4, j in {0..3}, k in {0..3}: A[4,j]*B[j,k] -> C[k,4]
        #    This is a 1x4 * 4x4 = 1x4 block, rank 16
        # 4. i=4, j in {0..3}, k=4: A[4,j]*B[j,4] -> C[4,4]
        #    This is a 1x4 * 4x1 = 1x1 block, rank 4
        # 5. i=4, j=4, k in {0..3}: A[4,4]*B[4,k] -> C[k,4]
        #    This is a 1x1 * 4x4 = 4x4 block, rank 16
        # 6. i=4, j=4, k=4: A[4,4]*B[4,4] -> C[4,4]
        #    This is a 1x1 * 1x1 = 1x1 block, rank 1
        
        # Total: 16+4+16+4+16+1 = 57
        # But we counted 61 above. Let me recount.
        
        # Actually, the 4x4x4 block covers i,j,k all in {0,1,2,3}
        # The remaining terms are those where at least one of i,j,k is 4
        # Total 125 - 64 = 61
        
        # Let me list them:
        # i=4, j in {0..4}, k in {0..4}: 25 terms
        # i in {0..3}, j=4, k in {0..4}: 20 terms
        # i in {0..3}, j in {0..3}, k=4: 16 terms
        # Total: 25+20+16 = 61. Correct.
        
        # Now, can we group these?
        # Group 1: i in {0..3}, j=4, k in {0..3}: 16 terms
        #   A[i,4]*B[4,k] -> C[k,i]
        #   This is like a 4x4 matrix product where A is 4x1 and B is 1x4
        #   Rank 16 (schoolbook)
        
        # Group 2: i in {0..3}, j=4, k=4: 4 terms
        #   A[i,4]*B[4,4] -> C[4,i]
        #   Rank 4 (schoolbook)
        
        # Group 3: i=4, j in {0..3}, k in {0..3}: 16 terms
        #   A[4,j]*B[j,k] -> C[k,4]
        #   Rank 16 (schoolbook)
        
        # Group 4: i=4, j in {0..3}, k=4: 4 terms
        #   A[4,j]*B[j,4] -> C[4,4]
        #   Rank 4 (schoolbook)
        
        # Group 5: i=4, j=4, k in {0..3}: 4 terms
        #   A[4,4]*B[4,k] -> C[k,4]
        #   Rank 4 (schoolbook)
        
        # Group 6: i=4, j=4, k=4: 1 term
        #   A[4,4]*B[4,4] -> C[4,4]
        #   Rank 1 (schoolbook)
        
        # Total: 16+4+16+4+4+1 = 45
        # But we need 61 terms. What's missing?
        
        # Ah, I forgot:
        # i in {0..3}, j in {0..3}, k=4: 16 terms
        #   A[i,j]*B[j,4] -> C[4,i]
        #   Rank 16 (schoolbook)
        
        # So total: 16+4+16+4+4+1+16 = 61. Correct.
        
        # Let's add these using schoolbook
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    # Skip if all indices are in {0,1,2,3} (already covered by 4x4x4)
                    if i < 4 and j < 4 and k < 4:
                        continue
                    # Use schoolbook
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[i*5 + j] = 1
                    v_vec[j*5 + k] = 1
                    w_vec[k*5 + i] = 1
                    add_col(u_vec, v_vec, w_vec)
        
        # Total rank: 49 + 61 = 110

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
