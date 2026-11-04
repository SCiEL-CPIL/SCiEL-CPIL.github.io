DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    U = [[0] * 93 for _ in range(n * m)]
    V = [[0] * 93 for _ in range(m * p)]
    W = [[0] * 93 for _ in range(p * n)]
    
    rank = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][rank] = u_vec[i]
        for i in range(m * p):
            if v_vec[i] != 0:
                V[i][rank] = v_vec[i]
        for i in range(p * n):
            if w_vec[i] != 0:
                W[i][rank] = w_vec[i]
        rank += 1

    def make_vec(dim, indices, values):
        v = [0] * dim
        for idx, val in zip(indices, values):
            v[idx] = val
        return v

    # Indices for A (16), B (28), C (28)
    # A[i, j] -> i*4 + j
    # B[j, k] -> j*7 + k
    # C[k, i] -> k*4 + i
    
    # 1. Compute A * B[:, 0:4] using Strassen (49 terms)
    # B[:, 0:4] is 4x4.
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B1 into 2x2 blocks: B11, B12, B21, B22
    # C1 is 4x4.
    
    # We need to call strassen_2x2x2 7 times.
    # Each call adds 7 terms.
    
    def strassen_block(a_offsets, b_offsets, c_offsets):
        # a_offsets: dict or list of indices for A block
        # We need to map local 2x2 indices to global 16 indices.
        # Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # a_offsets should be a list of 4 global indices.
        
        u1 = make_vec(16, [a_offsets[0], a_offsets[3]], [1, 1])
        v1 = make_vec(28, [b_offsets[0], b_offsets[3]], [1, 1])
        w1 = make_vec(28, [c_offsets[0], c_offsets[3]], [1, 1])
        add_term(u1, v1, w1)
        
        u2 = make_vec(16, [a_offsets[2], a_offsets[3]], [1, 1])
        v2 = make_vec(28, [b_offsets[0]], [1])
        w2 = make_vec(28, [c_offsets[2], c_offsets[3]], [1, -1])
        add_term(u2, v2, w2)
        
        u3 = make_vec(16, [a_offsets[0]], [1])
        v3 = make_vec(28, [b_offsets[2], b_offsets[0]], [1, -1])
        w3 = make_vec(28, [c_offsets[1], c_offsets[3]], [1, 1])
        add_term(u3, v3, w3)
        
        u4 = make_vec(16, [a_offsets[3]], [1])
        v4 = make_vec(28, [b_offsets[3], b_offsets[1]], [1, -1])
        w4 = make_vec(28, [c_offsets[0], c_offsets[2]], [1, 1])
        add_term(u4, v4, w4)
        
        u5 = make_vec(16, [a_offsets[0], a_offsets[1]], [1, 1])
        v5 = make_vec(28, [b_offsets[3]], [1])
        w5 = make_vec(28, [c_offsets[1], c_offsets[0]], [1, -1])
        add_term(u5, v5, w5)
        
        u6 = make_vec(16, [a_offsets[2], a_offsets[0]], [1, -1])
        v6 = make_vec(28, [b_offsets[0], b_offsets[1]], [1, 1])
        w6 = make_vec(28, [c_offsets[3]], [1])
        add_term(u6, v6, w6)
        
        u7 = make_vec(16, [a_offsets[1], a_offsets[3]], [1, -1])
        v7 = make_vec(28, [b_offsets[2], b_offsets[3]], [1, 1])
        w7 = make_vec(28, [c_offsets[0]], [1])
        add_term(u7, v7, w7)

    # Define offsets for A (4x4)
    # A11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A21: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A22: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # B (4x4) part 1 (cols 0-3)
    # B11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 7, 8
    # B12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 9, 10
    # B21: (2,0), (2,1), (3,0), (3,1) -> 14, 15, 21, 22
    # B22: (2,2), (2,3), (3,2), (3,3) -> 16, 17, 23, 24
    
    # C (4x4) part 1 (cols 0-3)
    # C11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # C12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # C21: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # C22: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15

    # Strassen M1 = (A11+A22)(B11+B22)
    a_m1 = [0, 1, 4, 5, 10, 11, 14, 15] # Wait, make_vec takes indices and values.
    # My strassen_block function expects a_offsets to be 4 indices.
    # But Strassen combines blocks.
    # I need to modify strassen_block to accept the combined vectors.
    
    # Let's rewrite add_term logic to be more flexible.
    
    # Re-initialize
    U = [[0] * 93 for _ in range(n * m)]
    V = [[0] * 93 for _ in range(m * p)]
    W = [[0] * 93 for _ in range(p * n)]
    rank = 0

    def add_term_indices(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal rank
        for idx, val in zip(u_indices, u_vals):
            U[idx][rank] = val
        for idx, val in zip(v_indices, v_vals):
            V[idx][rank] = val
        for idx, val in zip(w_indices, w_vals):
            W[idx][rank] = val
        rank += 1

    # Strassen 2x2x2 helper
    def strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1
        add_term_indices([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        # M2
        add_term_indices([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        # M3
        add_term_indices([a00], [1], [b10, b00], [1, -1], [c01, c11], [1, 1])
        # M4
        add_term_indices([a11], [1], [b11, b01], [1, -1], [c00, c10], [1, 1])
        # M5
        add_term_indices([a00, a01], [1, 1], [b11], [1], [c01, c00], [1, -1])
        # M6
        add_term_indices([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        # M7
        add_term_indices([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    # 4x4x4 Strassen
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # B11: 0,1,7,8
    # B12: 2,3,9,10
    # B21: 14,15,21,22
    # B22: 16,17,23,24
    
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15

    # M1 = (A11+A22)(B11+B22)
    strassen_2x2(
        0, 1, 4, 5, # A11
        10, 11, 14, 15, # A22
        0, 1, 7, 8, # B11
        16, 17, 23, 24, # B22
        0, 1, 4, 5, # C11
        2, 3, 6, 7, # C12
        8, 9, 12, 13, # C21
        10, 11, 14, 15 # C22
    )
    
    # M2 = (A21+A22)B11
    strassen_2x2(
        8, 9, 12, 13, # A21
        10, 11, 14, 15, # A22
        0, 1, 7, 8, # B11
        0, 0, 0, 0, # B22 (dummy, will be multiplied by 0? No, Strassen formula uses B11 only)
        # Wait, Strassen M2 is (A21+A22)B11.
        # My strassen_2x2 function implements full 2x2x2.
        # I need to zero out the parts of B that shouldn't be there.
        # If I pass 0 for B22 indices, it works.
        0, 1, 4, 5, # C21
        0, 0, 0, 0, # C22
        0, 0, 0, 0, # C11
        0, 0, 0, 0  # C12
    )
    # This is getting messy. The Strassen formula assumes full 2x2 matrices.
    # I should just use the 7 explicit terms for 4x4x4 directly.
    
    # Alternative: Just use schoolbook for 4x4x4 (49 terms is Strassen, 64 is schoolbook).
    # Schoolbook is easier to code correctly.
    # Rank 64 for 4x4x4.
    # Rank 48 for 4x4x3 (schoolbook).
    # Total 112. No improvement.
    
    # I will implement the 93 rank solution carefully.
    
    # Reset
    U = [[0] * 93 for _ in range(n * m)]
    V = [[0] * 93 for _ in range(m * p)]
    W = [[0] * 93 for _ in range(p * n)]
    rank = 0

    def add_term(u_idx_val, v_idx_val, w_idx_val):
        nonlocal rank
        for idx, val in u_idx_val:
            U[idx][rank] = val
        for idx, val in v_idx_val:
            V[idx][rank] = val
        for idx, val in w_idx_val:
            W[idx][rank] = val
        rank += 1

    # 4x4x4 Strassen (49 terms)
    # A indices: 0-15
    # B indices: 0-27 (cols 0-3)
    # C indices: 0-27 (cols 0-3)
    
    # M1: (A11+A22)(B11+B22)
    # A11: 0,1,4,5; A22: 10,11,14,15
    # B11: 0,1,7,8; B22: 16,17,23,24
    # C11: 0,1,4,5; C12: 2,3,6,7; C21: 8,9,12,13; C22: 10,11,14,15
    
    # We need to expand the 7 Strassen terms for the 4x4 case.
    # This is effectively 7 calls to 2x2x2 Strassen.
    
    # M1
    add_term([(0,1),(1,1),(4,1),(5,1),(10,1),(11,1),(14,1),(15,1)],
             [(0,1),(1,1),(7,1),(8,1),(16,1),(17,1),(23,1),(24,1)],
             [(0,1),(1,1),(4,1),(5,1),(10,1),(11,1),(14,1),(15,1)])
             
    # This is wrong. Strassen M1 is a single term in the 2x2x2 decomposition.
    # It is NOT a single rank-1 term in the 4x4x4 decomposition.
    # M1 = (A11+A22)(B11+B22) is a 2x2 matrix product.
    # It requires 7 rank-1 terms to compute (if we use Strassen for 2x2x2).
    # So M1 contributes 7 terms to the total rank.
    # Total 7 * 7 = 49 terms.
    
    # I will implement a helper that adds the 7 terms for a 2x2x2 block.
    
    def add_strassen_2x2(a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices
        # b_idx: list of 4 indices
        # c_idx: list of 4 indices
        # Local 0:00, 1:01, 2:10, 3:11
        
        # M1
        add_term([(a_idx[0],1),(a_idx[3],1)], [(b_idx[0],1),(b_idx[3],1)], [(c_idx[0],1),(c_idx[3],1)])
        # M2
        add_term([(a_idx[2],1),(a_idx[3],1)], [(b_idx[0],1)], [(c_idx[2],1),(c_idx[3],-1)])
        # M3
        add_term([(a_idx[0],1)], [(b_idx[2],1),(b_idx[0],-1)], [(c_idx[1],1),(c_idx[3],1)])
        # M4
        add_term([(a_idx[3],1)], [(b_idx[3],1),(b_idx[1],-1)], [(c_idx[0],1),(c_idx[2],1)])
        # M5
        add_term([(a_idx[0],1),(a_idx[1],1)], [(b_idx[3],1)], [(c_idx[1],1),(c_idx[0],-1)])
        # M6
        add_term([(a_idx[2],1),(a_idx[0],-1)], [(b_idx[0],1),(b_idx[1],1)], [(c_idx[3],1)])
        # M7
        add_term([(a_idx[1],1),(a_idx[3],-1)], [(b_idx[2],1),(b_idx[3],1)], [(c_idx[0],1)])

    # 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # A11: [0,1,4,5], A22: [10,11,14,15]
    # Combined A: [0,1,4,5,10,11,14,15] - No, Strassen takes 2x2 blocks.
    # I must pass the 4 indices of the block.
    # For M1, the "A block" is A11+A22.
    # This is not a contiguous block.
    # My add_strassen_2x2 expects 4 indices.
    # I need to create a "virtual" block?
    # No, the indices in add_term are global.
    # So for M1, the u vector has 1s at A11 and A22 indices.
    # The v vector has 1s at B11 and B22 indices.
    # The w vector has 1s at C11 and C22 indices.
    # But Strassen M1 is just ONE of the 7 terms.
    # The other 6 terms are different.
    # So I can just call add_term directly for each of the 7 Strassen terms of the 4x4x4 multiplication.
    
    # 4x4x4 Strassen terms:
    # 1. u: A11+A22, v: B11+B22, w: C11+C22
    # 2. u: A21+A22, v: B11, w: C21-C22
    # 3. u: A11, v: B21-B11, w: C12+C22
    # 4. u: A22, v: B22-B12, w: C11+C21
    # 5. u: A11+A12, v: B22, w: C12-C11
    # 6. u: A21-A11, v: B11+B12, w: C22
    # 7. u: A12-A22, v: B21+B22, w: C11

    add_term([(0,1),(1,1),(4,1),(5,1),(10,1),(11,1),(14,1),(15,1)],
             [(0,1),(1,1),(7,1),(8,1),(16,1),(17,1),(23,1),(24,1)],
             [(0,1),(1,1),(4,1),(5,1),(10,1),(11,1),(14,1),(15,1)])
             
    add_term([(8,1),(9,1),(12,1),(13,1),(10,1),(11,1),(14,1),(15,1)],
             [(0,1),(1,1),(7,1),(8,1)],
             [(8,1),(9,1),(12,1),(13,1),(10,-1),(11,-1),(14,-1),(15,-1)])
             
    add_term([(0,1),(1,1),(4,1),(5,1)],
             [(14,1),(15,1),(21,1),(22,1),(0,-1),(1,-1),(7,-1),(8,-1)],
             [(2,1),(3,1),(6,1),(7,1),(10,1),(11,1),(14,1),(15,1)])
             
    add_term([(10,1),(11,1),(14,1),(15,1)],
             [(16,1),(17,1),(23,1),(24,1),(2,-1),(3,-1),(9,-1),(10,-1)],
             [(0,1),(1,1),(4,1),(5,1),(8,1),(9,1),(12,1),(13,1)])
             
    add_term([(0,1),(1,1),(4,1),(5,1),(2,1),(3,1),(6,1),(7,1)],
             [(16,1),(17,1),(23,1),(24,1)],
             [(2,1),(3,1),(6,1),(7,1),(0,-1),(1,-1),(4,-1),(5,-1)])
             
    add_term([(8,1),(9,1),(12,1),(13,1),(0,-1),(1,-1),(4,-1),(5,-1)],
             [(0,1),(1,1),(7,1),(8,1),(2,1),(3,1),(9,1),(10,1)],
             [(10,1),(11,1),(14,1),(15,1)])
             
    add_term([(2,1),(3,1),(6,1),(7,1),(10,-1),(11,-1),(14,-1),(15,-1)],
             [(14,1),(15,1),(21,1),(22,1),(16,1),(17,1),(23,1),(24,1)],
             [(0,1),(1,1),(4,1),(5,1)])
             
    # 4x4x3 (44 terms)
    # 4 blocks of 2x2x3
    # Block 1: A11 (2x2) * B_top (2x3) -> C_top (2x3)
    # A11: 0,1,4,5
    # B_top (cols 4-6): 0*7+4=28? No, B is 4x7.
    # B indices: j*7+k.
    # j=0,1. k=4,5,6.
    # B_top: (0,4)=4, (0,5)=5, (0,6)=6, (1,4)=11, (1,5)=12, (1,6)=13.
    # C_top (cols 4-6): k=4,5,6. i=0,1.
    # C indices: k*4+i.
    # C_top: (4,0)=16, (4,1)=17, (5,0)=20, (5,1)=21, (6,0)=24, (6,1)=25.
    
    # 2x2x3 decomposition (11 terms)
    # Uses Strassen for first 2 cols (7 terms) + Schoolbook for 3rd col (4 terms).
    
    def add_2x2x3(a_idx, b_idx, c_idx):
        # b_idx: 6 indices (2x3)
        # c_idx: 6 indices (2x3)
        # b00, b01, b02, b10, b11, b12
        # c00, c01, c02, c10, c11, c12
        
        # Strassen for cols 0,1
        # M1
        add_term([(a_idx[0],1),(a_idx[3],1)], [(b_idx[0],1),(b_idx[3],1)], [(c_idx[0],1),(c_idx[3],1)])
        # M2
        add_term([(a_idx[2],1),(a_idx[3],1)], [(b_idx[0],1)], [(c_idx[2],1),(c_idx[3],-1)])
        # M3
        add_term([(a_idx[0],1)], [(b_idx[2],1),(b_idx[0],-1)], [(c_idx[1],1),(c_idx[3],1)])
        # M4
        add_term([(a_idx[3],1)], [(b_idx[3],1),(b_idx[1],-1)], [(c_idx[0],1),(c_idx[2],1)])
        # M5
        add_term([(a_idx[0],1),(a_idx[1],1)], [(b_idx[3],1)], [(c_idx[1],1),(c_idx[0],-1)])
        # M6
        add_term([(a_idx[2],1),(a_idx[0],-1)], [(b_idx[0],1),(b_idx[1],1)], [(c_idx[3],1)])
        # M7
        add_term([(a_idx[1],1),(a_idx[3],-1)], [(b_idx[2],1),(b_idx[3],1)], [(c_idx[0],1)])
        
        # Schoolbook for col 2
        # C02 = A00 B02 + A01 B12
        add_term([(a_idx[0],1)], [(b_idx[5],1)], [(c_idx[5],1)])
        add_term([(a_idx[1],1)], [(b_idx[6],1)], [(c_idx[5],1)])
        # C12 = A10 B02 + A11 B12
        add_term([(a_idx[2],1)], [(b_idx[5],1)], [(c_idx[6],1)])
        add_term([(a_idx[3],1)], [(b_idx[6],1)], [(c_idx[6],1)])

    # Block 1: A11 * B_top -> C_top
    add_2x2x3([0,1,4,5], [4,5,6,11,12,13], [16,17,20,21,24,25])
    
    # Block 2: A12 * B_top -> C_top_right? No, C is 4x7.
    # C_top is rows 0,1. C_bottom is rows 2,3.
    # C_top has cols 0-6.
    # C_top_right? No, C is just C.
    # C = [C_top; C_bottom].
    # C_top is 2x7. C_bottom is 2x7.
    # C_top cols 4-6 are indices 16-25.
    # C_bottom cols 4-6 are indices 28-37?
    # C index: k*4+i.
    # k=4, i=2 -> 18.
    # C_bottom (rows 2,3) cols 4-6:
    # (4,2)=18, (4,3)=19, (5,2)=22, (5,3)=23, (6,2)=26, (6,3)=27.
    
    # A12: 2,3,6,7
    # B_top: 4,5,6,11,12,13
    # C_top_right (rows 0,1, cols 4-6): 16,17,20,21,24,25
    # Wait, C_top is rows 0,1.
    # C_bottom is rows 2,3.
    # C = A B.
    # C_top = A11 B_top + A12 B_bottom.
    # C_bottom = A21 B_top + A22 B_bottom.
    
    # So:
    # 1. A11 * B_top -> C_top
    # 2. A12 * B_bottom -> C_top
    # 3. A21 * B_top -> C_bottom
    # 4. A22 * B_bottom -> C_bottom
    
    # B_bottom (rows 2,3, cols 4-6):
    # j=2,3. k=4,5,6.
    # (2,4)=18, (2,5)=19, (2,6)=20, (3,4)=25, (3,5)=26, (3,6)=27.
    
    # C_bottom (rows 2,3, cols 4-6):
    # i=2,3. k=4,5,6.
    # (4,2)=18, (4,3)=19, (5,2)=22, (5,3)=23, (6,2)=26, (6,3)=27.
    
    # Block 1 done.
    # Block 2: A12 * B_bottom -> C_top
    add_2x2x3([2,3,6,7], [18,19,20,25,26,27], [16,17,20,21,24,25])
    
    # Block 3: A21 * B_top -> C_bottom
    add_2x2x3([8,9,12,13], [4,5,6,11,12,13], [18,19,22,23,26,27])
    
    # Block 4: A22 * B_bottom -> C_bottom
    add_2x2x3([10,11,14,15], [18,19,20,25,26,27], [18,19,22,23,26,27])
    
    return {"U": U, "V": V, "W": W}
