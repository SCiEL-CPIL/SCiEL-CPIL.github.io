import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Rank 61 decomposition
    rank = 61
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_k, c_base_i, c_base_k):
        """
        Adds Strassen decomposition for A[a_base_i:a_base_i+2, a_base_j:a_base_j+2]
        and B[b_base_i:b_base_i+2, b_base_k:b_base_k+2]
        contributing to C[c_base_i:c_base_i+2, c_base_k:c_base_k+2]
        """
        # Local indices in 2x2 blocks
        # A: 0,0 -> a; 0,1 -> b; 1,0 -> c; 1,1 -> d
        # B: 0,0 -> e; 0,1 -> f; 1,0 -> g; 1,1 -> h
        
        a_idx = (a_base_i) * m + (a_base_j)
        b_idx = (a_base_i) * m + (a_base_j + 1)
        c_idx = (a_base_i + 1) * m + (a_base_j)
        d_idx = (a_base_i + 1) * m + (a_base_j + 1)
        
        e_idx = (b_base_i) * p + (b_base_k)
        f_idx = (b_base_i) * p + (b_base_k + 1)
        g_idx = (b_base_i + 1) * p + (b_base_k)
        h_idx = (b_base_i + 1) * p + (b_base_k + 1)
        
        # C output indices
        c00_idx = (c_base_k) * n + (c_base_i)
        c01_idx = (c_base_k + 1) * n + (c_base_i)
        c10_idx = (c_base_k) * n + (c_base_i + 1)
        c11_idx = (c_base_k + 1) * n + (c_base_i + 1)
        
        # m1 = (a+d)(e+h)
        add_term(a_idx, 1, e_idx, 1, c00_idx, 1)
        add_term(d_idx, 1, e_idx, 1, c11_idx, 1) # Wait, this adds two terms for one multiplication?
        
        # CORRECTION:
        # The tensor decomposition requires each *multiplication* to be a single column (rank 1 term)
        # if the U and V vectors are constant for that column.
        # But Strassen uses linear combinations.
        # For m1:
        # U vector has 1 at a, 1 at d.
        # V vector has 1 at e, 1 at h.
        # W vector has 1 at c00, 1 at c11.
        # This is ONE term in the rank decomposition.
        
        # My add_term function above adds ONE non-zero entry.
        # I need to modify the logic to build the column vector first, then append.
        
        # Let's restart the addition logic for Strassen.
        pass

    # Re-implementation of addition logic to handle multi-entry U,V,W vectors correctly.
    col = 0
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    def get_idx(i, j, dim1, dim2):
        return i * dim2 + j

    # 1. Strassen for A[0:2, 0:2] and B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1 (row 0), 3,4 (row 1) for cols 0,1
    # B indices: 0,1 (row 0), 7,8 (row 1) for cols 0,1
    # C indices: 0,1 (col 0), 3,4 (col 1) for rows 0,1
    
    a = [(0,0), (0,1), (1,0), (1,1)] # (i,j)
    b = [(0,0), (0,1), (1,0), (1,1)]
    
    # Map (i,j) to flat index for U, V, W
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i # W[k*n+i]
    
    # Strassen m1-m7 definitions for A(a,b;c,d) and B(e,f;g,h)
    # m1: (a+d)(e+h)
    U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
    U1[u_idx(0,0)] = 1; U1[u_idx(1,1)] = 1
    V1[v_idx(0,0)] = 1; V1[v_idx(1,1)] = 1
    W1[w_idx(0,0)] = 1; W1[w_idx(1,1)] = 1
    add_rank1_term(U1, V1, W1)
    
    # m2: (c+d)e
    U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
    U2[u_idx(1,0)] = 1; U2[u_idx(1,1)] = 1
    V2[v_idx(0,0)] = 1
    W2[w_idx(1,0)] = 1; W2[w_idx(1,1)] = -1
    add_rank1_term(U2, V2, W2)
    
    # m3: a(f-h)
    U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
    U3[u_idx(0,0)] = 1
    V3[v_idx(0,1)] = 1; V3[v_idx(1,1)] = -1
    W3[w_idx(0,1)] = 1; W3[w_idx(1,1)] = 1
    add_rank1_term(U3, V3, W3)
    
    # m4: d(g-e)
    U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
    U4[u_idx(1,1)] = 1
    V4[v_idx(1,0)] = 1; V4[v_idx(0,0)] = -1
    W4[w_idx(0,0)] = 1; W4[w_idx(1,0)] = 1
    add_rank1_term(U4, V4, W4)
    
    # m5: (a+b)h
    U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
    U5[u_idx(0,0)] = 1; U5[u_idx(0,1)] = 1
    V5[v_idx(1,1)] = 1
    W5[w_idx(0,0)] = -1; W5[w_idx(0,1)] = 1
    add_rank1_term(U5, V5, W5)
    
    # m6: (c-a)(e+f)
    U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
    U6[u_idx(1,0)] = 1; U6[u_idx(0,0)] = -1
    V6[v_idx(0,0)] = 1; V6[v_idx(0,1)] = 1
    W6[w_idx(1,1)] = 1
    add_rank1_term(U6, V6, W6)
    
    # m7: (b-d)(g+h)
    U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
    U7[u_idx(0,1)] = 1; U7[u_idx(1,1)] = -1
    V7[v_idx(1,0)] = 1; V7[v_idx(1,1)] = 1
    W7[w_idx(0,0)] = 1
    add_rank1_term(U7, V7, W7)
    
    # 2. Strassen for A[0:2, 0:2] and B[0:2, 2:4] -> C[0:2, 2:4]
    # A is same, B cols shifted by 2, C cols shifted by 2
    
    # m1
    U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
    U1[u_idx(0,0)] = 1; U1[u_idx(1,1)] = 1
    V1[v_idx(0,2)] = 1; V1[v_idx(1,3)] = 1
    W1[w_idx(0,2)] = 1; W1[w_idx(1,3)] = 1
    add_rank1_term(U1, V1, W1)
    
    # m2
    U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
    U2[u_idx(1,0)] = 1; U2[u_idx(1,1)] = 1
    V2[v_idx(0,2)] = 1
    W2[w_idx(1,2)] = 1; W2[w_idx(1,3)] = -1
    add_rank1_term(U2, V2, W2)
    
    # m3
    U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
    U3[u_idx(0,0)] = 1
    V3[v_idx(0,3)] = 1; V3[v_idx(1,3)] = -1
    W3[w_idx(0,3)] = 1; W3[w_idx(1,3)] = 1
    add_rank1_term(U3, V3, W3)
    
    # m4
    U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
    U4[u_idx(1,1)] = 1
    V4[v_idx(1,2)] = 1; V4[v_idx(0,2)] = -1
    W4[w_idx(0,2)] = 1; W4[w_idx(1,2)] = 1
    add_rank1_term(U4, V4, W4)
    
    # m5
    U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
    U5[u_idx(0,0)] = 1; U5[u_idx(0,1)] = 1
    V5[v_idx(1,3)] = 1
    W5[w_idx(0,2)] = -1; W5[w_idx(0,3)] = 1
    add_rank1_term(U5, V5, W5)
    
    # m6
    U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
    U6[u_idx(1,0)] = 1; U6[u_idx(0,0)] = -1
    V6[v_idx(0,2)] = 1; V6[v_idx(0,3)] = 1
    W6[w_idx(1,3)] = 1
    add_rank1_term(U6, V6, W6)
    
    # m7
    U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
    U7[u_idx(0,1)] = 1; U7[u_idx(1,1)] = -1
    V7[v_idx(1,2)] = 1; V7[v_idx(1,3)] = 1
    W7[w_idx(0,2)] = 1
    add_rank1_term(U7, V7, W7)
    
    # 3. Schoolbook for A[0:2, 0:2] and B[0:2, 4:7] -> C[0:2, 4:7]
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(i,j)] = 1
                V_term[v_idx(j,k)] = 1
                W_term[w_idx(i,k)] = 1
                add_rank1_term(U_term, V_term, W_term)
                
    # 4. Schoolbook for A[2, 0:2] and B[0:2, 0:7] -> C[2, 0:7]
    for j in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2,j)] = 1
            V_term[v_idx(j,k)] = 1
            W_term[w_idx(2,k)] = 1
            add_rank1_term(U_term, V_term, W_term)
            
    # 5. Schoolbook for A[0:3, 2] and B[2, 0:7] -> C[0:3, 0:7]
    for i in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i,2)] = 1
            V_term[v_idx(2,k)] = 1
            W_term[w_idx(i,k)] = 1
            add_rank1_term(U_term, V_term, W_term)
            
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Quick check
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook if error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
