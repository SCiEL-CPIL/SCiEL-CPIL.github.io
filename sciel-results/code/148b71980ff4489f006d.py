import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 110 (49 from 4x4 Strassen-Tensor + 61 from edge terms)
    rank = 110
    
    # Initialize U, V, W as lists of lists (matching seed style)
    # U: (n*m, rank) -> (25, 110)
    # V: (m*p, rank) -> (25, 110)
    # W: (p*n, rank) -> (25, 110)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 Strassen Tensor Product (Rank 49) ---
    
    # Strassen 2x2 coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = (a-b)(f-h)
    # p4 = d(g-h)
    # p5 = a(e+g)
    # p6 = (a+d)(h-g)
    # p7 = (c-b)(e+f)
    
    # U vectors for A (flattened 2x2: [a, b, c, d])
    U_strassen = [
        [1, 0, 1, 0], # a+d
        [0, 0, 1, 1], # c+d
        [1, -1, 0, 0],# a-b
        [0, 0, 0, 1], # d
        [1, 0, 0, 0], # a
        [1, 0, 1, 0], # a+d
        [0, -1, 1, 0] # c-b
    ]
    
    # V vectors for B (flattened 2x2: [e, f, g, h])
    V_strassen = [
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 1], # e+h
        [0, 1, 0, -1],# f-h
        [0, 0, 1, -1],# g-h
        [1, 0, 1, 0], # e+g
        [0, 0, -1, 1],# h-g
        [1, 1, 0, 0]  # e+f
    ]
    
    # W vectors for C (flattened 2x2: [C00, C01, C10, C11])
    # C00 = p1 - p2 + p4 + p6
    # C01 = p3 + p5
    # C10 = p4 + p7
    # C11 = p1 - p3 + p5 - p7
    W_strassen = [
        [1, 0, 0, 0],   # p1
        [-1, 0, 0, 0],  # p2
        [0, 1, 0, 0],   # p3
        [0, 0, 1, 1],   # p4
        [0, 1, 0, 1],   # p5
        [1, 0, 0, 0],   # p6
        [0, 0, 1, -1]   # p7
    ]
    # Note: The W vectors above represent the coefficients of each p_i in each C_ij.
    # For tensor product, we need the W part to be the Kronecker product of the outer and inner W structures.
    # Actually, the standard tensor product $T_1 \otimes T_2$ has:
    # U_{i,j} = U1_i \otimes U2_j
    # V_{i,j} = V1_i \otimes V2_j
    # W_{i,j} = W1_i \otimes W2_j
    
    col = 0
    for i in range(7):
        for j in range(7):
            # Kronecker product of U vectors
            # U1 is size 4, U2 is size 4. Result size 16.
            U_vec = np.kron(U_strassen[i], U_strassen[j])
            V_vec = np.kron(V_strassen[i], V_strassen[j])
            W_vec = np.kron(W_strassen[i], W_strassen[j])
            
            # Map 4x4 indices to 5x5 indices (0..3)
            # U is (5x5) flattened. 4x4 part is indices 0..15.
            # V is (5x5) flattened. 4x4 part is indices 0..15.
            # W is (5x5) flattened. 4x4 part is indices 0..15.
            # Wait, W index convention: W[k*n + i].
            # In the 4x4 sub-problem, the output index is (i, k) mapped to k*4 + i?
            # The seed code uses W[k*n + i].
            # For the 4x4 block embedded in 5x5, the local indices are 0..3.
            # The global W index for local (i, k) is k*5 + i?
            # No, the 4x4 block occupies rows/cols 0..3 of the 5x5 matrix.
            # So global i = local i, global k = local k.
            # Global W index = local_k * 5 + local_i.
            
            # However, the Kronecker product W_vec is size 16, corresponding to local (i, k) in 4x4.
            # Local W index in seed style for 4x4 would be k_local * 4 + i_local.
            # The Kronecker product of W vectors (which are 4-vectors) produces a 16-vector.
            # The order of np.kron(a, b) is a[0]*b, a[1]*b, ...
            # Let a = [A00, A01, A10, A11], b = [B00, B01, B10, B11].
            # np.kron(a,b) = [A00*B00, A00*B01, A00*B10, A00*B11, A01*B00, ...]
            # This corresponds to flattening the 4x4 matrix in row-major order (I, J) -> I*4+J.
            # But the seed W index is k*n + i (column-major style for output? No, k is row of B/col of C? 
            # Seed: C[i,k] uses W[k*n + i].
            # So W index is k * 5 + i.
            # This is column-major if we view (i,k) as (row, col)? No, i is row of C, k is col of C?
            # C = AB. C[i,k] = sum_j A[i,j]B[j,k].
            # So i is row index, k is column index.
            # W index = k * 5 + i.
            # This is column-major order of the C matrix (indexing by column k first, then row i).
            
            # The Kronecker product W_vec from np.kron is in row-major order of the 4x4 C matrix.
            # C_local[i, k].
            # We need to map C_local[i, k] to W_global[k * 5 + i].
            
            # Let's re-index W_vec.
            # W_vec[x] corresponds to local C[i_loc, k_loc] where x = i_loc * 4 + k_loc.
            # We need to place this value at global index k_loc * 5 + i_loc.
            
            for x in range(16):
                val = W_vec[x]
                if val != 0:
                    i_loc = x // 4
                    k_loc = x % 4
                    global_w_idx = k_loc * 5 + i_loc
                    W[global_w_idx][col] = val
            
            # U and V are just inputs, standard row-major flattening i*5+j.
            # The 4x4 block is top-left.
            # U_vec is row-major 4x4. x = i_loc * 4 + j_loc.
            # Global U idx = i_loc * 5 + j_loc.
            for x in range(16):
                val_u = U_vec[x]
                val_v = V_vec[x]
                if val_u != 0 or val_v != 0:
                    i_loc = x // 4
                    j_loc = x % 4
                    global_u_idx = i_loc * 5 + j_loc
                    global_v_idx = j_loc * 5 + i_loc # Wait, V is B[j,k].
                    # In 4x4, B is 4x4. Index is j_loc * 4 + k_loc?
                    # V_strassen is for B. Flattened B is [e,f,g,h] -> [B00, B01, B10, B11].
                    # So V_vec is row-major B.
                    # Global V idx = j_loc * 5 + k_loc?
                    # In the 5x5 case, B index is j * 5 + k.
                    # So global_v_idx = j_loc * 5 + k_loc?
                    # In the loop, x = i_loc * 4 + j_loc for U.
                    # For V, the corresponding index in the tensor product is also x.
                    # But V represents B. The indices are (j, k).
                    # In the 4x4 case, B is 4x4.
                    # So x = j_loc * 4 + k_loc.
                    # Global V idx = j_loc * 5 + k_loc.
                    
                    # Wait, np.kron(U_strassen[i], U_strassen[j]) aligns U1 and U2.
                    # U1 acts on the "outer" indices, U2 on "inner".
                    # The flattening is consistent.
                    
                    if val_u != 0:
                        U[global_u_idx][col] = val_u
                    if val_v != 0:
                        U_v_loc = j_loc
                        U_k_loc = k_loc
                        global_v_idx = j_loc * 5 + k_loc
                        V[global_v_idx][col] = val_v
                        
            col += 1
            
    # --- Part 2: Edge Terms (Rank 61) ---
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            raise Exception("Rank exceeded")
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Indices 0..3 are in 4x4 block. 4 is the 5th index.
    
    # 1. C[i,4] for i in 0..3: sum_{j=0..3} A[i,j] B[j,4] + A[i,4] B[4,4]
    for i in range(4):
        for j in range(4):
            # A[i,j] B[j,4] -> C[i,4]
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 5 + i # k=4, i=i
            add_term(u_idx, v_idx, w_idx)
            
        # A[i,4] B[4,4] -> C[i,4]
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        add_term(u_idx, v_idx, w_idx)

    # 2. C[4,k] for k in 0..3: sum_{j=0..3} A[4,j] B[j,k] + A[4,4] B[4,k]
    for k in range(4):
        for j in range(4):
            # A[4,j] B[j,k] -> C[4,k]
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            add_term(u_idx, v_idx, w_idx)
            
        # A[4,4] B[4,k] -> C[4,k]
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + k
        w_idx = k * 5 + 4
        add_term(u_idx, v_idx, w_idx)

    # 3. C[4,4]: sum_{j=0..3} A[4,j] B[j,4] + A[4,4] B[4,4]
    for j in range(4):
        u_idx = 4 * 5 + j
        v_idx = j * 5 + 4
        w_idx = 4 * 5 + 4
        add_term(u_idx, v_idx, w_idx)
        
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    add_term(u_idx, v_idx, w_idx)

    # 4. Correction C[i,k] for i,k in 0..3: A[i,4] B[4,k]
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 5 + i
            add_term(u_idx, v_idx, w_idx)

    return {"U": U, "V": V, "W": W}
