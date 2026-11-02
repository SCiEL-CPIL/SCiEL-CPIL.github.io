import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank) -> (12, rank)
    # V shape (m*p, rank) -> (24, rank)
    # W shape (p*n, rank) -> (18, rank)
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: list of (index, coeff)
        # v_vec: list of (index, coeff)
        # w_vec: list of (index, coeff)
        
        # We need to expand these into full vectors of size n*m, m*p, p*n
        # Or just keep them sparse and convert at the end?
        # The interface requires dense matrices.
        # So we will maintain dense lists.
        
        rank_so_far = len(U)
        
        # Initialize new columns
        U_new = [0] * (n * m)
        V_new = [0] * (m * p)
        W_new = [0] * (p * n)
        
        for idx, coeff in u_vec:
            U_new[idx] += coeff
        for idx, coeff in v_vec:
            V_new[idx] += coeff
        for idx, coeff in w_vec:
            W_new[idx] += coeff
            
        U.append(U_new)
        V.append(V_new)
        W.append(W_new)

    # Helper to get index in U (A matrix)
    # A is n x m. Index i, j.
    # Flat index: i * m + j
    def idx_A(i, j):
        return i * m + j

    # Helper to get index in V (B matrix)
    # B is m x p. Index j, k.
    # Flat index: j * p + k
    def idx_B(j, k):
        return j * p + k

    # Helper to get index in W (C matrix)
    # C is n x p. Index i, k.
    # Flat index: k * n + i  (Note: Task says W[k*n+i, r])
    def idx_C(i, k):
        return k * n + i

    # Strassen for 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def strassen_2x2(a_indices, b_indices, c_indices):
        # a_indices: dict or list of 4 indices for A
        # b_indices: dict or list of 4 indices for B
        # c_indices: dict or list of 4 indices for C
        
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # m1 = (a00 + a11)(b00 + b11)
        add_term([(a00, 1), (a11, 1)], 
                 [(b00, 1), (b11, 1)], 
                 [(c00, 1), (c11, 1)])
                 
        # m2 = (a10 + a11)b00
        add_term([(a10, 1), (a11, 1)], 
                 [(b00, 1)], 
                 [(c10, 1), (c11, -1)])
                 
        # m3 = a00(b01 - b11)
        add_term([(a00, 1)], 
                 [(b01, 1), (b11, -1)], 
                 [(c01, 1), (c11, 1)])
                 
        # m4 = a11(b10 - b00)
        add_term([(a11, 1)], 
                 [(b10, 1), (b00, -1)], 
                 [(c00, 1), (c10, 1)])
                 
        # m5 = (a00 + a01)b11
        add_term([(a00, 1), (a01, 1)], 
                 [(b11, 1)], 
                 [(c00, -1), (c01, 1)])
                 
        # m6 = (a10 - a00)(b00 + b01)
        add_term([(a10, 1), (a00, -1)], 
                 [(b00, 1), (b01, 1)], 
                 [(c11, 1)])
                 
        # m7 = (a01 - a11)(b10 + b11)
        add_term([(a01, 1), (a11, -1)], 
                 [(b10, 1), (b11, 1)], 
                 [(c00, 1), (c11, -1)])

    # Helper for 1x2xK dot products
    # A is 1x2: a0, a1
    # B is 2xK: b00..b0(K-1), b10..b1(K-1)
    # C is 1xK: c0..c(K-1)
    def dot_1x2(a_indices, b_indices, c_indices):
        a0, a1 = a_indices
        # b_indices is list of K pairs? Or flat list?
        # Let's pass b_indices as list of (j, k) or just flat indices.
        # Here B is 2xK.
        # Let's assume b_indices is a list of length K, where each element is a tuple (idx_top, idx_bot)?
        # No, let's just pass the flat indices for B and C.
        # B has 2*K entries.
        # C has K entries.
        
        # We need to map the structure.
        # Let's assume the caller passes the specific indices.
        
        pass # Replaced by inline logic below

    # We will process the two 3x2x6 blocks.
    # Block 1: A[:, 0:2] (3x2), B[0:2, :] (2x6)
    # Block 2: A[:, 2:4] (3x2), B[2:4, :] (2x6)
    
    for block in range(2):
        # A indices for this block:
        # Rows 0, 1, 2. Cols 2*block, 2*block+1.
        a_top_00 = idx_A(0, 2*block)
        a_top_01 = idx_A(0, 2*block+1)
        a_top_10 = idx_A(1, 2*block)
        a_top_11 = idx_A(1, 2*block+1)
        
        a_bot_0 = idx_A(2, 2*block)
        a_bot_1 = idx_A(2, 2*block+1)
        
        # B indices for this block:
        # Rows 2*block, 2*block+1. Cols 0..5.
        # B is 2x6.
        # Top row of B block: j = 2*block
        # Bot row of B block: j = 2*block+1
        
        # Split B into Left (cols 0-3) and Right (cols 4-5)
        
        # 1. A_top (2x2) * B_left (2x4)
        # B_left cols 0,1,2,3.
        # This is two 2x2 multiplications: B[:, 0:2] and B[:, 2:4].
        
        # Sub-block 1.1: B[:, 0:2]
        b_00 = idx_B(2*block, 0)
        b_01 = idx_B(2*block, 1)
        b_10 = idx_B(2*block+1, 0)
        b_11 = idx_B(2*block+1, 1)
        
        c_00 = idx_C(0, 0)
        c_01 = idx_C(0, 1)
        c_10 = idx_C(1, 0)
        c_11 = idx_C(1, 1)
        
        strassen_2x2([a_top_00, a_top_01, a_top_10, a_top_11], 
                     [b_00, b_01, b_10, b_11], 
                     [c_00, c_01, c_10, c_11])
                     
        # Sub-block 1.2: B[:, 2:4]
        b_00 = idx_B(2*block, 2)
        b_01 = idx_B(2*block, 3)
        b_10 = idx_B(2*block+1, 2)
        b_11 = idx_B(2*block+1, 3)
        
        c_00 = idx_C(0, 2)
        c_01 = idx_C(0, 3)
        c_10 = idx_C(1, 2)
        c_11 = idx_C(1, 3)
        
        strassen_2x2([a_top_00, a_top_01, a_top_10, a_top_11], 
                     [b_00, b_01, b_10, b_11], 
                     [c_00, c_01, c_10, c_11])

        # 2. A_top (2x2) * B_right (2x2)
        # B_right cols 4,5.
        b_00 = idx_B(2*block, 4)
        b_01 = idx_B(2*block, 5)
        b_10 = idx_B(2*block+1, 4)
        b_11 = idx_B(2*block+1, 5)
        
        c_00 = idx_C(0, 4)
        c_01 = idx_C(0, 5)
        c_10 = idx_C(1, 4)
        c_11 = idx_C(1, 5)
        
        strassen_2x2([a_top_00, a_top_01, a_top_10, a_top_11], 
                     [b_00, b_01, b_10, b_11], 
                     [c_00, c_01, c_10, c_11])

        # 3. A_bot (1x2) * B_left (2x4)
        # 8 terms.
        for k in range(4):
            # c_k = a_bot_0 * b_top_k + a_bot_1 * b_bot_k
            # b_top_k is idx_B(2*block, k)
            # b_bot_k is idx_B(2*block+1, k)
            # c_k is idx_C(2, k)
            
            add_term([(a_bot_0, 1)], 
                     [(idx_B(2*block, k), 1)], 
                     [(idx_C(2, k), 1)])
                     
            add_term([(a_bot_1, 1)], 
                     [(idx_B(2*block+1, k), 1)], 
                     [(idx_C(2, k), 1)])

        # 4. A_bot (1x2) * B_right (2x2)
        # 4 terms.
        for k in range(4, 6):
            add_term([(a_bot_0, 1)], 
                     [(idx_B(2*block, k), 1)], 
                     [(idx_C(2, k), 1)])
                     
            add_term([(a_bot_1, 1)], 
                     [(idx_B(2*block+1, k), 1)], 
                     [(idx_C(2, k), 1)])

    # Convert lists of lists to numpy arrays?
    # The task says "factor matrices of shapes ... with integer/half-integer entries".
    # Lists are probably fine, but numpy is safer for "matrix" implication.
    # However, the seed program returned lists.
    # Let's return lists to match the seed.
    
    return {"U": U, "V": V, "W": W}
