DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W as lists of lists (or arrays)
    # We don't know the final rank, so we build lists of rows.
    # U rows: n*m
    # V rows: m*p
    # W rows: p*n
    
    # We will accumulate columns.
    # Each column in U, V, W corresponds to a rank-1 term.
    # We'll store them as lists of vectors.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def zero_vec(size):
        return [0] * size
        
    def schoolbook_block(i_idx, j_idx, k_idx):
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    u = zero_vec(n*m)
                    v = zero_vec(m*p)
                    w = zero_vec(p*n)
                    
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    
                    add_term(u, v, w)
                    
    def strassen_block(i_idx, j_idx, k_idx):
        # i_idx, j_idx, k_idx are lists of 2 indices
        i0, i1 = i_idx
        j0, j1 = j_idx
        k0, k1 = k_idx
        
        # Map local indices to global U, V, W indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        u_a00 = i0*m + j0
        u_a01 = i0*m + j1
        u_a10 = i1*m + j0
        u_a11 = i1*m + j1
        
        v_b00 = j0*p + k0
        v_b01 = j0*p + k1
        v_b10 = j1*p + k0
        v_b11 = j1*p + k1
        
        w_c00 = k0*n + i0
        w_c01 = k1*n + i0
        w_c10 = k0*n + i1
        w_c11 = k1*n + i1
        
        # Define the 7 Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        u1 = zero_vec(n*m); u1[u_a00]=1; u1[u_a11]=1
        v1 = zero_vec(m*p); v1[v_b00]=1; v1[v_b11]=1
        w1 = zero_vec(p*n); w1[w_c00]=1; w1[w_c11]=1
        add_term(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = zero_vec(n*m); u2[u_a10]=1; u2[u_a11]=1
        v2 = zero_vec(m*p); v2[v_b00]=1
        w2 = zero_vec(p*n); w2[w_c10]=1; w2[w_c11]=-1
        add_term(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        u3 = zero_vec(n*m); u3[u_a00]=1
        v3 = zero_vec(m*p); v3[v_b10]=1; v3[v_b11]=-1
        w3 = zero_vec(p*n); w3[w_c01]=1; w3[w_c11]=1
        add_term(u3, v3, w3)
        
        # M4 = A11(B01 - B00)
        u4 = zero_vec(n*m); u4[u_a11]=1
        v4 = zero_vec(m*p); v4[v_b01]=1; v4[v_b00]=-1
        w4 = zero_vec(p*n); w4[w_c00]=1; w4[w_c10]=1
        add_term(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = zero_vec(n*m); u5[u_a00]=1; u5[u_a01]=1
        v5 = zero_vec(m*p); v5[v_b11]=1
        w5 = zero_vec(p*n); w5[w_c00]=-1; w5[w_c01]=1
        add_term(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = zero_vec(n*m); u6[u_a10]=1; u6[u_a00]=-1
        v6 = zero_vec(m*p); v6[v_b00]=1; v6[v_b01]=1
        w6 = zero_vec(p*n); w6[w_c11]=1
        add_term(u6, v6, w6)
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = zero_vec(n*m); u7[u_a01]=1; u7[u_a11]=-1
        v7 = zero_vec(m*p); v7[v_b10]=1; v7[v_b11]=1
        w7 = zero_vec(p*n); w7[w_c00]=1
        add_term(u7, v7, w7)
        
    # Define blocks
    # J partition: [0,1], [2,3], [4]
    # K partition: [0,1], [2,3], [4]
    
    # Strassen blocks (2x2 in J and K)
    strassen_block([0, 1], [0, 1], [0, 1])
    strassen_block([0, 1], [2, 3], [2, 3])
    strassen_block([2, 3], [0, 1], [0, 1])
    strassen_block([2, 3], [2, 3], [2, 3])
    
    # Schoolbook blocks for remaining parts
    # J=[4], K=[0,1]
    schoolbook_block([0, 1], [4], [0, 1])
    # J=[4], K=[2,3]
    schoolbook_block([0, 1], [4], [2, 3])
    # J=[4], K=[4]
    schoolbook_block([0, 1], [4], [4])
    # J=[0,1], K=[4]
    schoolbook_block([0, 1], [0, 1], [4])
    # J=[2,3], K=[4]
    schoolbook_block([0, 1], [2, 3], [4])
    
    # Convert lists of columns to matrices U, V, W
    # U is (n*m) x rank
    # We have U_cols as list of vectors of length n*m
    # We need to transpose
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for r in range(rank):
        u_vec = U_cols[r]
        v_vec = V_cols[r]
        w_vec = W_cols[r]
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
