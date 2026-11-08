"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition strategy.
    # M_{3,4,8} is decomposed into 8 blocks of M_{3,2,2}.
    # Each M_{3,2,2} is decomposed into M_{2,2,2} (rank 7) and M_{1,2,2} (rank 4).
    # Total rank = 8 * (7 + 4) = 88.
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Strassen decomposition for 2x2x2
    # U, V, W are lists of vectors
    # M_{2,2,2}
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W for the whole tensor.
    # Indices:
    # A: i in 0..2, j in 0..3
    # B: j in 0..3, k in 0..7
    # C: i in 0..2, k in 0..7
    
    # We map the 8 blocks to (j_pair, k_pair)
    # j_pair: 0 (j=0,1), 1 (j=2,3)
    # k_pair: 0 (k=0,1), 1 (k=2,3), 2 (k=4,5), 3 (k=6,7)
    
    total_rank = 0
    U = zeros(n * m, 0)
    V = zeros(m * p, 0)
    W = zeros(p * n, 0)
    
    def add_tensor(U_add, V_add, W_add):
        nonlocal total_rank
        # U_add: list of lists, shape (n*m, rank_add)
        # V_add: list of lists, shape (m*p, rank_add)
        # W_add: list of lists, shape (p*n, rank_add)
        rank_add = len(U_add[0]) if U_add else 0
        # Extend U, V, W
        for r in range(rank_add):
            U_col = [U_add[i][r] for i in range(n * m)]
            V_col = [V_add[i][r] for i in range(m * p)]
            W_col = [W_add[i][r] for i in range(p * n)]
            U.append(U_col)
            V.append(V_col)
            W.append(W_col)
        total_rank += rank_add
        
    # Wait, U, V, W are defined as rows. The interface expects (n*m, rank).
    # So U is a list of n*m rows, each of length rank.
    # My add_tensor logic above appends columns? No, it appends rows?
    # "factor matrices of shapes (n*m, rank)"
    # So U[i][c] is the entry.
    # I initialized U as zeros(n*m, 0).
    # I need to append columns to U, V, W.
    
    # Let's re-initialize to allow column appending.
    U = [ [] for _ in range(n * m) ]
    V = [ [] for _ in range(m * p) ]
    W = [ [] for _ in range(p * n) ]
    
    def append_column(U_col, V_col, W_col):
        for i in range(n * m):
            U[i].append(U_col[i])
        for i in range(m * p):
            V[i].append(V_col[i])
        for i in range(p * n):
            W[i].append(W_col[i])
            
    # Define M_{2,2,2} Strassen components
    # Local indices for 2x2 block:
    # A rows 0,1; A cols 0,1
    # B rows 0,1; B cols 0,1
    # C rows 0,1; C cols 0,1
    
    # We will generate the 7 terms for Strassen.
    # Each term is a rank-1 tensor.
    # Term t: U_t, V_t, W_t.
    
    def get_strassen_term(t, offset_i, offset_j, offset_k):
        # offset_i: row offset in A/C
        # offset_j: col offset in A / row offset in B
        # offset_k: col offset in B/C
        
        # A indices: i in {0,1} + offset_i, j in {0,1} + offset_j
        # B indices: j in {0,1} + offset_j, k in {0,1} + offset_k
        # C indices: i in {0,1} + offset_i, k in {0,1} + offset_k
        
        U_vec = [0] * (n * m)
        V_vec = [0] * (m * p)
        W_vec = [0] * (p * n)
        
        # Strassen formulas
        if t == 0: # M1 = (A11 + A22)(B11 + B22)
            # A11: i=0, j=0; A22: i=1, j=1
            U_vec[(offset_i + 0) * m + (offset_j + 0)] = 1
            U_vec[(offset_i + 1) * m + (offset_j + 1)] = 1
            # B11: j=0, k=0; B22: j=1, k=1
            V_vec[(offset_j + 0) * p + (offset_k + 0)] = 1
            V_vec[(offset_j + 1) * p + (offset_k + 1)] = 1
            # C11: i=0, k=0; C22: i=1, k=1
            W_vec[(offset_k + 0) * n + (offset_i + 0)] = 1
            W_vec[(offset_k + 1) * n + (offset_i + 1)] = 1
        elif t == 1: # M2 = (A21 + A22)B11
            U_vec[(offset_i + 1) * m + (offset_j + 0)] = 1
            U_vec[(offset_i + 1) * m + (offset_j + 1)] = 1
            V_vec[(offset_j + 0) * p + (offset_k + 0)] = 1
            W_vec[(offset_k + 0) * n + (offset_i + 1)] = 1
        elif t == 2: # M3 = A11(B12 - B22)
            U_vec[(offset_i + 0) * m + (offset_j + 0)] = 1
            V_vec[(offset_j + 0) * p + (offset_k + 1)] = 1
            V_vec[(offset_j + 1) * p + (offset_k + 1)] = -1
            W_vec[(offset_k + 1) * n + (offset_i + 0)] = 1
        elif t == 3: # M4 = A22(B21 - B11)
            U_vec[(offset_i + 1) * m + (offset_j + 1)] = 1
            V_vec[(offset_j + 1) * p + (offset_k + 0)] = 1
            V_vec[(offset_j + 0) * p + (offset_k + 0)] = -1
            W_vec[(offset_k + 0) * n + (offset_i + 1)] = 1
        elif t == 4: # M5 = (A11 + A12)B22
            U_vec[(offset_i + 0) * m + (offset_j + 0)] = 1
            U_vec[(offset_i + 0) * m + (offset_j + 1)] = 1
            V_vec[(offset_j + 1) * p + (offset_k + 1)] = 1
            W_vec[(offset_k + 1) * n + (offset_i + 0)] = 1
        elif t == 5: # M6 = (A21 - A11)(B11 + B12)
            U_vec[(offset_i + 1) * m + (offset_j + 0)] = 1
            U_vec[(offset_i + 0) * m + (offset_j + 0)] = -1
            V_vec[(offset_j + 0) * p + (offset_k + 0)] = 1
            V_vec[(offset_j + 0) * p + (offset_k + 1)] = 1
            W_vec[(offset_k + 0) * n + (offset_i + 1)] = 1
        elif t == 6: # M7 = (A12 - A22)(B21 + B22)
            U_vec[(offset_i + 0) * m + (offset_j + 1)] = 1
            U_vec[(offset_i + 1) * m + (offset_j + 1)] = -1
            V_vec[(offset_j + 1) * p + (offset_k + 0)] = 1
            V_vec[(offset_j + 1) * p + (offset_k + 1)] = 1
            W_vec[(offset_k + 1) * n + (offset_i + 0)] = 1
            
        # Apply C reconstruction formulas
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The W_vec above is just the "multiplication" part.
        # We need to combine these 7 terms to form the final W vectors.
        # Actually, it's easier to generate the 7 rank-1 tensors for the multiplications,
        # and then sum them up for W.
        # But the interface requires U, V, W to be the factors of the tensor.
        # The tensor is sum over c of U_c \otimes V_c \otimes W_c.
        # Strassen says:
        # C = sum_{t=0}^6 (U_t \otimes V_t \otimes W_t)
        # where U_t, V_t are the inputs to the t-th multiplication,
        # and W_t is the output placement.
        
        # Wait, the standard decomposition is:
        # C = sum_{t=0}^6 M_t
        # M_t = (U_t A) (V_t B) W_t
        # This is not quite right.
        # The rank-1 tensors are:
        # T_t = U_t \otimes V_t \otimes W_t
        # where U_t is a vector in A-space, V_t in B-space, W_t in C-space.
        # And C = sum T_t(A, B).
        
        # For Strassen:
        # T_0: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) -> C11, C22
        # T_1: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,0) -> C21
        # T_2: U=(1,0,0,0), V=(0,1,-1,0), W=(0,1,0,0) -> C12
        # T_3: U=(0,0,0,1), V=(0,0,1,-1), W=(0,0,1,0) -> C21
        # T_4: U=(1,1,0,0), V=(0,0,0,1), W=(0,1,0,0) -> C12
        # T_5: U=(0,-1,1,0), V=(1,1,0,0), W=(0,0,1,0) -> C21
        # T_6: U=(0,1,-1,0), V=(0,0,1,1), W=(1,0,0,0) -> C11
        
        # Let's verify T_0:
        # A dot U_0 = A11 + A22
        # B dot V_0 = B11 + B22
        # Product = M1
        # M1 contributes to C11 and C22.
        # W_0 has 1 at C11 and C22.
        # So T_0(A,B) = (A11+A22)(B11+B22) * (E11 + E22)
        # This adds M1 to C11 and M1 to C22. Correct.
        
        # So we can just define the 7 vectors for each block.
        
        return U_vec, V_vec, W_vec

    # This approach is getting messy with offsets.
    # Let's just build the full U, V, W matrices for the 8 blocks.
    
    U = [ [] for _ in range(n * m) ]
    V = [ [] for _ in range(m * p) ]
    W = [ [] for _ in range(p * n) ]
    
    def add_rank1(U_vec, V_vec, W_vec):
        for i in range(n * m):
            U[i].append(U_vec[i])
        for i in range(m * p):
            V[i].append(V_vec[i])
        for i in range(p * n):
            W[i].append(W_vec[i])
            
    # Strassen vectors for 2x2
    # A indices: 0,1,2,3 (r0c0, r0c1, r1c0, r1c1)
    # B indices: 0,1,2,3 (r0c0, r0c1, r1c0, r1c1)
    # C indices: 0,1,2,3 (r0c0, r0c1, r1c0, r1c1)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, -1, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, -1, 0], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1 -> C11, C22
        [0, 0, 1, 0], # M2 -> C21
        [0, 1, 0, 0], # M3 -> C12
        [0, 0, 1, 0], # M4 -> C21
        [0, 1, 0, 0], # M5 -> C12
        [0, 0, 1, 0], # M6 -> C21
        [1, 0, 0, 0]  # M7 -> C11
    ]
    
    # Schoolbook vectors for 1x2x2
    # A: 1x2 -> indices 0,1
    # B: 2x2 -> indices 0,1,2,3
    # C: 1x2 -> indices 0,1
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # T0: A0 B00 -> C0
    # T1: A1 B10 -> C0
    # T2: A0 B01 -> C1
    # T3: A1 B11 -> C1
    
    SB_U = [
        [1, 0], # T0
        [0, 1], # T1
        [1, 0], # T2
        [0, 1]  # T3
    ]
    SB_V = [
        [1, 0, 0, 0], # T0
        [0, 0, 1, 0], # T1
        [0, 1, 0, 0], # T2
        [0, 0, 0, 1]  # T3
    ]
    SB_W = [
        [1, 0], # T0 -> C0
        [1, 0], # T1 -> C0
        [0, 1], # T2 -> C1
        [0, 1]  # T3 -> C1
    ]
    
    # Now assemble the 8 blocks.
    # Block (jp, kp)
    # jp: 0, 1. Offset j = jp * 2.
    # kp: 0, 1, 2, 3. Offset k = kp * 2.
    
    for jp in range(2):
        for kp in range(4):
            offset_j = jp * 2
            offset_k = kp * 2
            
            # Part 1: M_{2,2,2} for rows 0,1
            # Rows 0,1 of A.
            # Cols offset_j, offset_j+1 of A.
            # Rows offset_j, offset_j+1 of B.
            # Cols offset_k, offset_k+1 of B.
            # Cols offset_k, offset_k+1 of C.
            # Rows 0,1 of C.
            
            for t in range(7):
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                # Fill U_vec for rows 0,1
                # A indices: i in {0,1}, j in {offset_j, offset_j+1}
                # Map local 0,1,2,3 to global
                # Local 0: i=0, j=offset_j
                # Local 1: i=0, j=offset_j+1
                # Local 2: i=1, j=offset_j
                # Local 3: i=1, j=offset_j+1
                
                local_u = S_U[t]
                U_vec[0 * m + offset_j] = local_u[0]
                U_vec[0 * m + offset_j + 1] = local_u[1]
                U_vec[1 * m + offset_j] = local_u[2]
                U_vec[1 * m + offset_j + 1] = local_u[3]
                
                # Fill V_vec
                # B indices: j in {offset_j, offset_j+1}, k in {offset_k, offset_k+1}
                # Local 0: j=offset_j, k=offset_k
                # Local 1: j=offset_j, k=offset_k+1
                # Local 2: j=offset_j+1, k=offset_k
                # Local 3: j=offset_j+1, k=offset_k+1
                
                local_v = S_V[t]
                V_vec[offset_j * p + offset_k] = local_v[0]
                V_vec[offset_j * p + offset_k + 1] = local_v[1]
                V_vec[(offset_j + 1) * p + offset_k] = local_v[2]
                V_vec[(offset_j + 1) * p + offset_k + 1] = local_v[3]
                
                # Fill W_vec
                # C indices: i in {0,1}, k in {offset_k, offset_k+1}
                # Local 0: i=0, k=offset_k
                # Local 1: i=0, k=offset_k+1
                # Local 2: i=1, k=offset_k
                # Local 3: i=1, k=offset_k+1
                
                local_w = S_W[t]
                W_vec[offset_k * n + 0] = local_w[0]
                W_vec[(offset_k + 1) * n + 0] = local_w[1]
                W_vec[offset_k * n + 1] = local_w[2]
                W_vec[(offset_k + 1) * n + 1] = local_w[3]
                
                add_rank1(U_vec, V_vec, W_vec)
                
            # Part 2: M_{1,2,2} for row 2
            # Row 2 of A.
            # Cols offset_j, offset_j+1 of A.
            # Rows offset_j, offset_j+1 of B.
            # Cols offset_k, offset_k+1 of B.
            # Col offset_k, offset_k+1 of C.
            # Row 2 of C.
            
            for t in range(4):
                U_vec = [0] * (n * m)
                V_vec = [0] * (m * p)
                W_vec = [0] * (p * n)
                
                # Fill U_vec for row 2
                local_u = SB_U[t]
                U_vec[2 * m + offset_j] = local_u[0]
                U_vec[2 * m + offset_j + 1] = local_u[1]
                
                # Fill V_vec
                local_v = SB_V[t]
                V_vec[offset_j * p + offset_k] = local_v[0]
                V_vec[offset_j * p + offset_k + 1] = local_v[1]
                V_vec[(offset_j + 1) * p + offset_k] = local_v[2]
                V_vec[(offset_j + 1) * p + offset_k + 1] = local_v[3]
                
                # Fill W_vec
                local_w = SB_W[t]
                W_vec[offset_k * n + 2] = local_w[0]
                W_vec[(offset_k + 1) * n + 2] = local_w[1]
                
                add_rank1(U_vec, V_vec, W_vec)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
