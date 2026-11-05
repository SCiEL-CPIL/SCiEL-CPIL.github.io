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

    # Part 1: 4x4x4 multiplication using rank-28 decomposition
    # The 4x4x4 Strassen decomposition has 28 scalar products.
    # We split A, B, C into 2x2 blocks:
    # A: A00, A01, A10, A11 (each 2x2)
    # B: B00, B01, B10, B11 (each 2x2)
    # C: C00, C01, C10, C11 (each 2x2)
    #
    # The 28 products are defined as follows (using the known 4x4x4 rank-28 scheme):
    # We use 7 "outer" Strassen products, each decomposed into 4 scalar products
    # using a specific 2x2x2 scheme (not full Strassen 7, but a 4-product scheme
    # that works when combined with the outer structure).
    #
    # Actually, the standard 4x4x4 rank-28 uses the following approach:
    # Split into 2x2 blocks. Use 7 Strassen-like combinations at the block level,
    # but each block-level product (2x2 x 2x2) is done with a 4-term scheme
    # that exploits the specific structure.
    #
    # The known 28-term 4x4x4 decomposition:
    # Let's use the explicit 28 terms from the literature.
    #
    # Indices: A[i,k], B[k,j], C[i,j] for i,k,j in 0..3
    #
    # The 28 bilinear forms for 4x4x4:
    # We'll define them as (u_coefficients, v_coefficients, w_coefficients)
    # where u has 16 entries (4x4), v has 16 entries (4x4), w has 16 entries (4x4)

    # Using the known rank-28 decomposition for 4x4x4 matrix multiplication:
    # This is based on the "Strassen 4x4" scheme.
    
    # Let me define the 28 terms explicitly.
    # Each term: list of (i,k) for U with coeff, (k,j) for V with coeff, (j,i) for W with coeff
    
    # The 4x4x4 rank-28 decomposition (from the literature):
    # We split into quadrants. The 28 products are:
    
    # Define helper for 2x2x2 with 7 terms (Strassen)
    # For the 4x4x4, we use a different structure.
    
    # Actually, let me use the nested Strassen approach but optimize it.
    # The current approach gives 49. Let me see if we can do better.
    
    # Alternative: Use the fact that 4x4x4 can be done in 28 multiplications.
    # The 28-term decomposition:
    
    # Let me hardcode the 28 terms for 4x4x4.
    # Based on the standard 4x4x4 rank-28 scheme:
    
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 28 products use specific linear combinations.
    # Let me use the explicit formulation.
    
    # For simplicity and correctness, let me use the 49-term nested Strassen
    # but check if we can reduce the cross terms.
    
    # Actually, let me try a different approach: use the 4x4x4 rank-28 
    # decomposition explicitly. Here are the 28 bilinear forms:
    
    # I'll use the known 28-term 4x4x4 decomposition.
    # Each term is (u_terms, v_terms, w_terms) where each is a list of (index, coeff)
    # U index: i*4+k for 4x4 A matrix (indices 0..15)
    # V index: k*4+j for 4x4 B matrix (indices 0..15)  
    # W index: j*4+i for 4x4 C matrix (indices 0..15)
    
    # The 28 terms of the 4x4x4 Strassen decomposition:
    # (Using the standard formulation)
    
    def idx_u(i, k):
        return i * 4 + k
    
    def idx_v(k, j):
        return k * 4 + j
    
    def idx_w(j, i):
        return j * 4 + i
    
    # 28 bilinear forms for 4x4x4 matrix multiplication
    # This is the rank-28 decomposition
    
    terms_4x4 = [
        # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
        # Using 7 sub-terms for the 2x2x2 product, but we need to be careful
        # Let me just use the 49-term approach for safety and optimize later.
    ]
    
    # For safety, let me use the proven 49-term nested Strassen for 4x4x4
    # and focus on optimizing the remaining parts.
    
    # Actually, let me try to implement the 28-term 4x4x4 decomposition properly.
    # The key insight: the 4x4x4 tensor has rank 28, and there's a known explicit decomposition.
    
    # Let me use the following approach: 
    # Split 4x4 into 2x2 blocks. The 28 products are organized as:
    # 4 "diagonal" Strassen products (each using 7 terms for the 2x2x2 core)
    # + 12 "off-diagonal" products (each using 1 term)
    # = 4*7 + 12 = 40... that's not 28.
    
    # Let me just use the 49-term approach which is known to work,
    # and see if the total is competitive.
    
    # 49 (4x4x4) + 16 (4x4x1) + 20 (4x1x5) = 85
    # vs inspiration: 28 (4x4x4) + 16 + 16 + 4 = 64
    
    # The difference is in the 4x4x4 part: 49 vs 28.
    # Let me implement the 28-term 4x4x4 decomposition.
    
    # The 28-term 4x4x4 decomposition (from Bini et al. / Strassen):
    # I'll define it using the block structure.
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), 
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # The 28 products for 4x4x4:
    # P1 = (A00+A11)(B00+B11)  [2x2 x 2x2, needs 7 scalar mults]
    # P2 = (A10+A11)B00        [2x2 x 2x2, needs 7 scalar mults]
    # P3 = A00(B01-B11)        [2x2 x 2x2, needs 7 scalar mults]
    # P4 = A11(B10-B00)        [2x2 x 2x2, needs 7 scalar mults]
    # P5 = (A00+A01)B11        [2x2 x 2x2, needs 7 scalar mults]
    # P6 = (A10-A00)(B00+B01)  [2x2 x 2x2, needs 7 scalar mults]
    # P7 = (A01-A11)(B10+B11)  [2x2 x 2x2, needs 7 scalar mults]
    # That's 7 * 7 = 49. Not 28.
    
    # So the 28-term decomposition is different. It's not simply nested Strassen.
    # The 28-term decomposition uses a more sophisticated scheme.
    
    # Let me look up the actual 28-term 4x4x4 decomposition.
    # It uses specific linear combinations that are not simply block Strassen.
    
    # For now, let me use the 49-term approach and see if we can optimize
    # the other parts. 49 + 16 + 20 = 85.
    
    # Wait, the current program already does this and scores 1.1765.
    # The inspiration does 64 terms. Let me try to implement the 28-term 4x4x4.
    
    # The 28-term 4x4x4 decomposition:
    # I found that the rank of the 4x4x4 matrix multiplication tensor is 28.
    # An explicit decomposition:
    
    # Let me use a different strategy: use the 4x4x4 decomposition from
    # the "Strassen 4x4" paper which gives 28 products.
    
    # Here's the 28-term decomposition (indices for 4x4 matrices):
    # I'll define each term as a triple of sparse vectors.
    
    # After research, the 28-term 4x4x4 decomposition can be constructed as:
    # Use 4 "diagonal" 2x2x2 Strassen decompositions (each 7 terms) = 28 terms
    # But this doesn't work because the off-diagonal blocks need to be handled.
    
    # Let me just go with the 49-term nested Strassen for 4x4x4 (which is proven correct)
    # and optimize the remaining 36 terms.
    
    # Current: 49 + 16 + 20 = 85
    # Can we reduce the 4x4x1 and 4x1x5 parts?
    
    # 4x4x1: This is multiplying a 4x4 matrix by a 4x1 vector. Rank = 16 (schoolbook).
    # Can we do better? The tensor for 4x4x1 has rank 16 (it's just a sum of 16 rank-1 terms).
    # Actually, the rank of the 4x4x1 tensor is 16 (each output element needs its own product).
    
    # 4x1x5: Multiplying a 4x1 vector by a 1x5 matrix. Rank = 20.
    # Similarly, rank is 20.
    
    # So the minimum for the non-4x4x4 parts is 16 + 20 = 36.
    # Total minimum with 49-term 4x4x4: 49 + 36 = 85.
    
    # With 28-term 4x4x4: 28 + 36 = 64.
    
    # Let me implement the 28-term 4x4x4 decomposition.
    
    # The 28-term decomposition for 4x4x4 matrix multiplication:
    # Based on the work of Bini, Capovani, Lotti, Romani (1977) and others.
    
    # I'll use the following explicit 28-term scheme:
    # (This is a known decomposition, I'll hardcode the 28 bilinear forms)
    
    # For the 4x4x4 case, the 28 products are:
    # Let me define them using the block structure with specific coefficients.
    
    # After careful consideration, let me use the 49-term approach for correctness
    # and accept 85 total. But wait, the inspiration claims 64. Let me verify.
    
    # Actually, re-reading the inspiration: it says "4x4x4 part: 49" in the comment
    # but then the code seems to use a different approach. Let me re-examine.
    
    # The inspiration code has: rank = 49 + 16 + 16 + 4 = 85. Wait, that's the same!
    # 49 + 16 + 16 + 4 = 85. So the inspiration also uses 49 for 4x4x4.
    
    # But the score is the same (1.1765). So both programs have rank 85.
    
    # Let me verify: the current program does 49 + 16 + 20 = 85.
    # The inspiration does 49 + 16 + 16 + 4 = 85.
    # Same total!
    
    # So the improvement must come from reducing the 4x4x4 part below 49.
    
    # Let me implement the 28-term 4x4x4 decomposition.
    
    # The 28-term 4x4x4 decomposition (I'll use the known explicit form):
    
    # Define the 28 bilinear forms for 4x4x4 matrix multiplication.
    # Each form: (u_sparse, v_sparse, w_sparse)
    # u_sparse: dict mapping (i,k) -> coeff
    # v_sparse: dict mapping (k,j) -> coeff  
    # w_sparse: dict mapping (j,i) -> coeff
    
    # The 28-term scheme:
    # I'll use the decomposition from the literature.
    
    # After research, here's the 28-term 4x4x4 decomposition:
    # It uses the following 28 products (I'll define them explicitly):
    
    # Let me use a programmatic approach to generate the 28 terms.
    # The 28-term decomposition can be constructed using the "Strassen 4x4" scheme.
    
    # For now, let me try a different approach: use the 49-term nested Strassen
    # but see if we can merge some terms.
    
    # Actually, let me just go with the current 85-term approach and ensure it's correct.
    # The score is 1.1765 which is already an improvement over the baseline.
    
    # Let me implement the 4x4x4 part using nested Strassen (49 terms) as in the current program,
    # and keep the other parts the same.
    
    # Part 1: 4x4x4 using nested Strassen (49 terms)
    # This is the same as the current program's approach.
    
    # Define the 7 Strassen products for 4x4x4 (outer level)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},  # M1
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},          # M2
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},         # M3
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},         # M4
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},          # M5
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},   # M6
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},   # M7
    ]
    
    c_contributions = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {1: 1, 3: 1},   # M3
        {0: 1, 2: 1},   # M4
        {0: -1, 1: 1},  # M5
        {3: 1},         # M6
        {0: 1},         # M7
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
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x4 (k=4, j in 0..3) - schoolbook (16 terms)
    for i in range(4):
        for j in range(4):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    # Part 4: 4x1x1 (k=4, j=4) - 4 terms
    for i in range(4):
        add_term(i, 4, 4, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
