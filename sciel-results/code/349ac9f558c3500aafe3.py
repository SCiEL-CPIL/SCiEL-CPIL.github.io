import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    # We'll build the decomposition incrementally
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        if time.time() > deadline:
            return False
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        return True

    # Helper: create Strassen terms for A[a_i:a_i+2, a_j:a_j+2] * B[b_i:b_i+2, b_k:b_k+2] -> C[c_i:c_i+2, c_k:c_k+2]
    def add_strassen(a_i, a_j, b_i, b_k, c_i, c_k):
        # A entries: (a_i, a_j)=a, (a_i, a_j+1)=b, (a_i+1, a_j)=c, (a_i+1, a_j+1)=d
        # B entries: (b_i, b_k)=e, (b_i, b_k+1)=f, (b_i+1, b_k)=g, (b_i+1, b_k+1)=h
        # C entries: (c_i, c_k)=c00, (c_i, c_k+1)=c01, (c_i+1, c_k)=c10, (c_i+1, c_k+1)=c11
        
        a_i, a_j, b_i, b_k, c_i, c_k = int(a_i), int(a_j), int(b_i), int(b_k), int(c_i), int(c_k)
        
        # m1: (a+d)(e+h) -> c00 + c11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_i, b_k)] = 1; V1[v_idx(b_i+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        if not add_rank1_term(U1, V1, W1): return False
        
        # m2: (c+d)*e -> c10 - c11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_i, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        if not add_rank1_term(U2, V2, W2): return False
        
        # m3: a*(f-h) -> c01 + c11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_i, b_k+1)] = 1; V3[v_idx(b_i+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        if not add_rank1_term(U3, V3, W3): return False
        
        # m4: d*(g-e) -> c00 + c10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_i+1, b_k)] = 1; V4[v_idx(b_i, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        if not add_rank1_term(U4, V4, W4): return False
        
        # m5: (a+b)*h -> -c00 + c01
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_i+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        if not add_rank1_term(U5, V5, W5): return False
        
        # m6: (c-a)*(e+f) -> c11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_i, b_k)] = 1; V6[v_idx(b_i, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        if not add_rank1_term(U6, V6, W6): return False
        
        # m7: (b-d)*(g+h) -> c00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_i+1, b_k)] = 1; V7[v_idx(b_i+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        if not add_rank1_term(U7, V7, W7): return False
        
        return True

    # Helper: schoolbook term
    def add_schoolbook(i, j, k):
        U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
        U_t[u_idx(i, j)] = 1
        V_t[v_idx(j, k)] = 1
        W_t[w_idx(i, k)] = 1
        return add_rank1_term(U_t, V_t, W_t)

    # Strategy: 
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (Strassen: 7)
    ok = add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (Strassen: 7)
    ok = ok and add_strassen(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (Schoolbook 2x2x1: 4)
    for i in range(2):
        for j in range(2):
            ok = ok and add_schoolbook(i, j, 4)
    
    # Block 4: A[0:2,0:2] * B[0:2,5:7] -> C[0:2,5:7]  (Strassen: 7)
    ok = ok and add_strassen(0, 0, 0, 5, 0, 5)
    
    # Block 5: A[2,0:2] * B[0:2,0:7] -> C[2,0:7]  (Schoolbook 1x2x7: 14)
    for j in range(2):
        for k in range(7):
            ok = ok and add_schoolbook(2, j, k)
    
    # Block 6: A[0:3,2] * B[2,0:7] -> C[0:3,0:7]  (Schoolbook 3x1x7: 21)
    for i in range(3):
        for k in range(7):
            ok = ok and add_schoolbook(i, 2, k)

    # Total: 7 + 7 + 4 + 7 + 14 + 21 = 60
    rank = len(U_cols)
    
    if not ok or rank == 0:
        # Fallback to schoolbook
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
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            if U_cols[r][i] != 0:
                U[i][r] = U_cols[r][i]
        for i in range(m * p):
            if V_cols[r][i] != 0:
                V[i][r] = V_cols[r][i]
        for i in range(p * n):
            if W_cols[r][i] != 0:
                W[i][r] = W_cols[r][i]
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
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
        # Fallback to schoolbook
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
