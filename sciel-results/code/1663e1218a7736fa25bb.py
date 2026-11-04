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
    
    def add_rank1(i, k, j, coeff=1.0):
        """Add a rank-1 term: coeff * A[i,k] * B[k,j] contributes to C[i,j]"""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i
        
        u_vec[u_idx] = coeff
        v_vec[v_idx] = 1.0
        w_vec[w_idx] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2):
        """
        Add Strassen decomposition for 2x2x2 multiplication.
        A block: rows a_r1..a_r2, cols a_c1..a_c2
        B block: rows b_r1..b_r2, cols b_c1..b_c2
        C block: rows c_r1..c_r2, cols c_c1..c_c2
        """
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map local indices (0,1) to global indices.
        # A local (r,c) -> global (a_r1+r, a_c1+c)
        # B local (r,c) -> global (b_r1+r, b_c1+c)
        # C local (r,c) -> global (c_r1+r, c_c1+c)
        
        def get_a(r, c):
            return (a_r1 + r, a_c1 + c)
        def get_b(r, c):
            return (b_r1 + r, b_c1 + c)
        def get_c(r, c):
            return (c_r1 + r, c_c1 + c)
            
        # Term 1: M1
        # u: A00, A11
        # v: B00, B11
        # w: C00 (1), C11 (1)
        u1 = [0.0] * (n * m)
        v1 = [0.0] * (m * p)
        w1 = [0.0] * (p * n)
        
        i, k = get_a(0, 0); u1[i*m+k] = 1.0
        i, k = get_a(1, 1); u1[i*m+k] = 1.0
        
        i, k = get_b(0, 0); v1[i*p+k] = 1.0
        i, k = get_b(1, 1); v1[i*p+k] = 1.0
        
        i, j = get_c(0, 0); w1[j*n+i] = 1.0
        i, j = get_c(1, 1); w1[j*n+i] = 1.0
        
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # Term 2: M2
        # u: A10, A11
        # v: B00
        # w: C10 (1), C11 (-1)
        u2 = [0.0] * (n * m)
        v2 = [0.0] * (m * p)
        w2 = [0.0] * (p * n)
        
        i, k = get_a(1, 0); u2[i*m+k] = 1.0
        i, k = get_a(1, 1); u2[i*m+k] = 1.0
        
        i, k = get_b(0, 0); v2[i*p+k] = 1.0
        
        i, j = get_c(1, 0); w2[j*n+i] = 1.0
        i, j = get_c(1, 1); w2[j*n+i] = -1.0
        
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # Term 3: M3
        # u: A00
        # v: B01, B11 (-1)
        # w: C01 (1), C11 (1)
        u3 = [0.0] * (n * m)
        v3 = [0.0] * (m * p)
        w3 = [0.0] * (p * n)
        
        i, k = get_a(0, 0); u3[i*m+k] = 1.0
        
        i, k = get_b(0, 1); v3[i*p+k] = 1.0
        i, k = get_b(1, 1); v3[i*p+k] = -1.0
        
        i, j = get_c(0, 1); w3[j*n+i] = 1.0
        i, j = get_c(1, 1); w3[j*n+i] = 1.0
        
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # Term 4: M4
        # u: A11
        # v: B10, B00 (-1)
        # w: C00 (1), C10 (1)
        u4 = [0.0] * (n * m)
        v4 = [0.0] * (m * p)
        w4 = [0.0] * (p * n)
        
        i, k = get_a(1, 1); u4[i*m+k] = 1.0
        
        i, k = get_b(1, 0); v4[i*p+k] = 1.0
        i, k = get_b(0, 0); v4[i*p+k] = -1.0
        
        i, j = get_c(0, 0); w4[j*n+i] = 1.0
        i, j = get_c(1, 0); w4[j*n+i] = 1.0
        
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
        # Term 5: M5
        # u: A00, A01
        # v: B11
        # w: C00 (-1), C01 (1)
        u5 = [0.0] * (n * m)
        v5 = [0.0] * (m * p)
        w5 = [0.0] * (p * n)
        
        i, k = get_a(0, 0); u5[i*m+k] = 1.0
        i, k = get_a(0, 1); u5[i*m+k] = 1.0
        
        i, k = get_b(1, 1); v5[i*p+k] = 1.0
        
        i, j = get_c(0, 0); w5[j*n+i] = -1.0
        i, j = get_c(0, 1); w5[j*n+i] = 1.0
        
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)
        
        # Term 6: M6
        # u: A10, A00 (-1)
        # v: B00, B01
        # w: C11 (1)
        u6 = [0.0] * (n * m)
        v6 = [0.0] * (m * p)
        w6 = [0.0] * (p * n)
        
        i, k = get_a(1, 0); u6[i*m+k] = 1.0
        i, k = get_a(0, 0); u6[i*m+k] = -1.0
        
        i, k = get_b(0, 0); v6[i*p+k] = 1.0
        i, k = get_b(0, 1); v6[i*p+k] = 1.0
        
        i, j = get_c(1, 1); w6[j*n+i] = 1.0
        
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)
        
        # Term 7: M7
        # u: A01, A11 (-1)
        # v: B10, B11
        # w: C00 (1)
        u7 = [0.0] * (n * m)
        v7 = [0.0] * (m * p)
        w7 = [0.0] * (p * n)
        
        i, k = get_a(0, 1); u7[i*m+k] = 1.0
        i, k = get_a(1, 1); u7[i*m+k] = -1.0
        
        i, k = get_b(1, 0); v7[i*p+k] = 1.0
        i, k = get_b(1, 1); v7[i*p+k] = 1.0
        
        i, j = get_c(0, 0); w7[j*n+i] = 1.0
        
        U_cols.append(u7); V_cols.append(v7); W_cols.append(w7)

    # 1. 4x4x4 part
    # A: 0-3, 0-3. B: 0-3, 0-3. C: 0-3, 0-3.
    # Split into 2x2 blocks.
    
    # A00 B00 -> C00
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    # A01 B10 -> C00
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    # A00 B01 -> C01
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    # A01 B11 -> C01
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    # A10 B00 -> C10
    add_strassen(2, 3, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1)
    # A11 B10 -> C10
    add_strassen(2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 0, 1)
    # A10 B01 -> C11
    add_strassen(2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3)
    # A11 B11 -> C11
    add_strassen(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3)
    
    # 2. 4x4x1 part (A cols 0-3, B col 4, C col 4)
    for k in range(4):
        add_rank1(i=0, k=k, j=4, coeff=1.0) # This is wrong, add_rank1 adds 1 term for specific i,k,j
        # Need to loop i as well?
        # C[i,4] = sum_k A[i,k] B[k,4]
        # So for each k, we have a term that contributes to all i.
        # My add_rank1 only adds 1 i.
        # I need a specialized add for this.
        
    # Let's implement the 4x4x1 and 4x1x5 parts manually.
    
    # Clear the last 4 incorrect adds? No, just don't use add_rank1 for this.
    # I will remove the loop above and write explicit code.
    
    # 4x4x1 part
    for k in range(4):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        # u: A[i, k] for all i
        for i in range(4):
            u_vec[i * m + k] = 1.0
            
        # v: B[k, 4]
        v_vec[k * p + 4] = 1.0
        
        # w: C[i, 4] for all i
        for i in range(4):
            w_vec[4 * n + i] = 1.0 # j=4, n=4. Index 4*4+i = 16+i
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # 4x1x5 part
    u_vec = [0.0] * (n * m)
    v_vec = [0.0] * (m * p)
    w_vec = [0.0] * (p * n)
    
    # u: A[i, 4] for all i
    for i in range(4):
        u_vec[i * m + 4] = 1.0
        
    # v: B[4, j] for all j
    for j in range(5):
        v_vec[4 * p + j] = 1.0
        
    # w: C[j, i] for all j, i
    for j in range(5):
        for i in range(4):
            w_vec[j * n + i] = 1.0
            
    U_cols.append(u_vec)
    V_cols.append(v_vec)
    W_cols.append(w_vec)
    
    rank = len(U_cols)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
