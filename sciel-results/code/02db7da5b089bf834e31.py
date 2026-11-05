import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        
        add_rank1(u_vec, v_vec, w_vec)
        
    # Part 1: 4x4x4 multiplication (k in 0..3, j in 0..3) using Recursive Strassen
    # Big Strassen (4x4)
    strassen_products_big = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    
    c_contributions_big = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    # Small Strassen (2x2)
    # Local indices 0,1 correspond to the 2x2 block elements
    strassen_products_small = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}},
        {'A': {(0,0): 1}, 'B': {(1,0): 1, (0,0): -1}},
        {'A': {(1,1): 1}, 'B': {(0,1): 1, (1,0): -1}},
        {'A': {(0,0): 1, (1,0): 1}, 'B': {(1,1): 1}},
        {'A': {(0,1): 1, (0,0): -1}, 'B': {(0,0): 1, (1,0): 1}},
        {'A': {(1,0): 1, (1,1): -1}, 'B': {(1,1): 1, (0,1): 1}},
    ]
    
    c_contributions_small = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): -1}, # M2 (Note: C10 gets M2, C11 gets M2? No, see below)
        # Let's re-verify Small Strassen C contributions
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2 - M5 - M7
        # C11 = M2 + M3 + M6
        
        # So:
        # M1: C00+=1, C10+=1
        # M2: C10+=1, C11+=1
        # M3: C01+=1, C11+=1
        # M4: C00-=1, C01+=1
        # M5: C00+=1, C10-=1
        # M6: C11+=1
        # M7: C00+=1, C10-=1
        
        # Let's define c_contributions_small correctly
    ]
    
    c_contributions_small = [
        {(0,0): 1, (1,0): 1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,1): 1, (1,1): 1}, # M3
        {(0,0): -1, (0,1): 1}, # M4
        {(0,0): 1, (1,0): -1}, # M5
        {(1,1): 1}, # M6
        {(0,0): 1, (1,0): -1}, # M7
    ]
    
    def a_global(block, li, lk):
        if block == 0:
            return li, lk
        elif block == 1:
            return li, lk + 2
        elif block == 2:
            return li + 2, lk
        elif block == 3:
            return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0:
            return lk, lj
        elif block == 1:
            return lk, lj + 2
        elif block == 2:
            return lk + 2, lj
        elif block == 3:
            return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        elif block == 3:
            return li + 2, lj + 2
            
    for big_idx, big_prod in enumerate(strassen_products_big):
        a_form_big = big_prod['A']
        b_form_big = big_prod['B']
        c_contrib_big = c_contributions_big[big_idx]
        
        for small_idx, small_prod in enumerate(strassen_products_small):
            a_form_small = small_prod['A']
            b_form_small = small_prod['B']
            c_contrib_small = c_contributions_small[small_idx]
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            # Combine Big and Small A forms
            # A_global = sum_{block} sign_big * (sum_{local} sign_small * A_local)
            # We need to map local (li, lk) to global (gi, gk) via the block.
            # The block is determined by the Big Strassen form.
            # For a given Big Strassen product, it involves specific blocks.
            # For each block involved, we apply the Small Strassen form.
            
            # It is easier to iterate over the blocks in the Big form
            # and for each block, iterate over the local elements in the Small form.
            
            # However, the Big form gives signs for blocks.
            # The Small form gives signs for local elements.
            # The total sign is sign_big * sign_small.
            
            # We must be careful: The Big Strassen algorithm computes products of 2x2 blocks.
            # The Small Strassen algorithm computes products of scalars within those 2x2 blocks.
            # So for each Big Product (which is a sum of Block products),
            # we expand each Block product into 7 Scalar products.
            
            # So:
            # Term = sum_{block_a, block_b} sign_big_a * sign_big_b * (Product of Block_a and Block_b)
            # Product of Block_a and Block_b = sum_{small} sign_small_a * sign_small_b * (Product of Local_a and Local_b)
            
            # We iterate over the blocks present in the Big form.
            # But the Big form is a dictionary of block -> sign.
            # It implies that the Big Product is a linear combination of Block Products.
            # e.g. M1 = (A0+A3)(B0+B3) = A0B0 + A0B3 + A3B0 + A3B3.
            # This is a sum of 4 Block Products.
            # Each Block Product is a 2x2x2 tensor.
            # We decompose each 2x2x2 tensor into 7 rank-1 terms.
            # So M1 contributes 4 * 7 = 28 terms?
            # No, Strassen M1 is ONE product of two 2x2 matrices.
            # The rank of the 2x2x2 tensor is 7.
            # So M1 (one 2x2 product) is decomposed into 7 rank-1 terms.
            # The expression (A0+A3)(B0+B3) is a single 2x2 matrix product.
            # So we just apply the 7 Small Strassen terms to this combined block product.
            
            # Wait, the "Block" in the Big Strassen context is a 2x2 submatrix.
            # The "Local" in the Small Strassen context is a scalar element of that 2x2 submatrix.
            
            # So for a specific Big Product (e.g. M1), it defines a 2x2 matrix A_M1 and B_M1.
            # A_M1 is a linear combination of the 2x2 blocks.
            # B_M1 is a linear combination of the 2x2 blocks.
            # The product A_M1 * B_M1 is a 2x2 matrix.
            # We decompose this 2x2 multiplication into 7 rank-1 terms.
            
            # So, for each Big Product:
            # 1. Construct the effective 2x2 A and B matrices (in terms of global indices).
            # 2. Apply the 7 Small Strassen patterns.
            
            # Let's implement this.
            
            # For each Small Strassen pattern (local indices):
            # We need to sum the contributions from all (block_a, block_b) pairs that contribute to this Big Product?
            # No. The Big Product IS the 2x2 matrix.
            # The Small Strassen patterns operate on the elements of this 2x2 matrix.
            
            # So, for a fixed Big Product (big_idx) and Small Product (small_idx):
            # The term corresponds to:
            # U: sum over (block_a, local_a) in small_prod.A of sign_small_a * (sum over block_a in big_prod.A of sign_big_a * A_global(block_a, local_a))
            # This is getting complex to nest.
            
            # Alternative:
            # The Big Strassen algorithm generates 7 products of 2x2 matrices.
            # Let P_r be the r-th 2x2 product.
            # P_r = A_r * B_r.
            # A_r and B_r are 2x2 matrices formed by linear combinations of the 4 blocks.
            # We decompose P_r into 7 rank-1 terms using Small Strassen.
            # So for each r (0..6), we add 7 terms.
            # Total 49 terms.
            
            # How to construct the vectors for the r-th, s-th term?
            # The s-th Small Strassen term for P_r involves specific local elements of A_r and B_r.
            # A_r's local element (li, lk) is a linear combination of global elements A[gi, gk].
            # Specifically, A_r[li, lk] = sum_{block} sign_big[block] * A[gi(block, li), gk(block, lk)].
            
            # So, for a fixed big_idx and small_idx:
            # u_vec is zero everywhere.
            # For each local index (li, lk) in small_prod['A']:
            #   sign_s = small_prod['A'][(li, lk)]
            #   For each block in big_prod['A']:
            #     sign_b = big_prod['A'][block]
            #     gi, gk = a_global(block, li, lk)
            #     u_vec[gi * m + gk] += sign_b * sign_s
            
            # Similarly for v_vec and w_vec.
            
            for (li, lk), sign_s_a in a_form_small.items():
                for block_a, sign_b_a in a_form_big.items():
                    gi, gk = a_global(block_a, li, lk)
                    u_vec[gi * m + gk] += sign_b_a * sign_s_a
            
            for (lk, lj), sign_s_b in b_form_small.items():
                for block_b, sign_b_b in b_form_big.items():
                    gk, gj = b_global(block_b, lk, lj)
                    v_vec[gk * p + gj] += sign_b_b * sign_s_b
            
            # For W, the contribution is to the C matrix of the Big Product.
            # The Big Product contributes to specific C blocks with signs from c_contrib_big.
            # The Small Product contributes to specific local C elements with signs from c_contrib_small.
            # The total C element is C[gi, gj].
            # C[gi, gj] is formed by summing over Big Blocks and Small Local elements.
            # But the Big Strassen algorithm says:
            # C = sum_r (C_r * P_r) where C_r is the 2x2 matrix of coefficients.
            # So for a fixed r (big_idx), the contribution to C is C_r * P_r.
            # P_r is the 2x2 product.
            # The s-th Small Strassen term approximates part of P_r.
            # So the term adds to C_r's elements.
            
            # Wait, the decomposition is:
            # P_r = sum_s U_s^r \otimes V_s^r \otimes W_s^r
            # Then C = sum_r C_r \otimes P_r = sum_r C_r \otimes (sum_s ...) = sum_r sum_s (C_r \otimes U_s^r \otimes V_s^r \otimes W_s^r)
            # This is not a simple sum of rank-1 tensors in the original space?
            # Actually, yes.
            # The term is:
            # U_part: from A_r
            # V_part: from B_r
            # W_part: from C_r element
            
            # So for fixed big_idx (r) and small_idx (s):
            # u_vec is constructed from A_r as above.
            # v_vec is constructed from B_r as above.
            # w_vec:
            # For each local (li, lj) in c_contrib_small:
            #   sign_s_c = c_contrib_small[(li, lj)]
            #   For each block_c in c_contrib_big:
            #     sign_b_c = c_contrib_big[block_c]
            #     gi, gj = c_global(block_c, li, lj)
            #     w_vec[gj * n + gi] += sign_b_c * sign_s_c
            
            for (li, lj), sign_s_c in c_contrib_small.items():
                for block_c, sign_b_c in c_contrib_big.items():
                    gi, gj = c_global(block_c, li, lj)
                    w_vec[gj * n + gi] += sign_b_c * sign_s_c
            
            add_rank1(u_vec, v_vec, w_vec)
            
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook
    for i in range(4):
        for k in range(4):
            if time.time() - _START > _DEADLINE - 10:
                break
            add_term(i, k, 4, 1.0, 1.0, 1.0)
            
    # Part 3: 4x1x5 (k=4, j in 0..4) - Rank 1 term
    u_vec = [0.0] * (n * m)
    v_vec = [0.0] * (m * p)
    w_vec = [0.0] * (p * n)
    for i in range(4):
        u_vec[i * m + 4] = 1.0
    for j in range(5):
        v_vec[4 * p + j] = 1.0
    for j in range(5):
        for i in range(4):
            w_vec[j * n + i] = 1.0
    add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
        
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
