"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # C_bot (1x5x6): 30
    # C_top (2x5x6):
    #   A11 B11 (2x2x6) -> 3 * 7 = 21
    #   A12 B12 (2x3x6) -> 3 * (7 + 4) = 33
    # Total: 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        # This interface is tricky because one term can have multiple non-zero entries in U, V, W
        # It is better to define the term by its U, V, W vectors.
        pass

    # Better approach: define the rank-1 components explicitly.
    # Each rank-1 component is a tuple (U_vec, V_vec, W_vec).
    # We will build a list of these tuples and then populate the matrices.
    
    terms = []
    
    # 1. C_bot: 1x5x6
    # i = 2
    for j in range(5):
        for k in range(6):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            u_vec[2 * 5 + j] = 1
            v_vec[j * 6 + k] = 1
            w_vec[k * 3 + 2] = 1
            
            terms.append((u_vec, v_vec, w_vec))
            
    # 2. C_top: 2x5x6
    # Split j into 0,1 (A11) and 2,3,4 (A12)
    
    # Helper for Strassen 2x2x2
    def get_strassen_terms(a_indices, b_indices, c_indices):
        # a_indices: list of (u_idx) for a,b,c,d
        # b_indices: list of (v_idx) for e,f,g,h
        # c_indices: list of (w_idx) for c00, c01, c10, c11
        # Returns list of (u_vec, v_vec, w_vec)
        
        # a,b,c,d are the 4 entries of A (2x2)
        # e,f,g,h are the 4 entries of B (2x2)
        # c00,c01,c10,c11 are the 4 entries of C (2x2)
        
        a_idx, b_idx, c_idx, d_idx = a_indices
        e_idx, f_idx, g_idx, h_idx = b_indices
        c00_idx, c01_idx, c10_idx, c11_idx = c_indices
        
        res = []
        
        # M1: (a+d)(e+h) -> C00 +, C11 +
        u1 = [0]*(n*m); u1[a_idx]=1; u1[d_idx]=1
        v1 = [0]*(m*p); v1[e_idx]=1; v1[h_idx]=1
        w1 = [0]*(p*n); w1[c00_idx]=1; w1[c11_idx]=1
        res.append((u1, v1, w1))
        
        # M2: (c+d)e -> C10 +
        u2 = [0]*(n*m); u2[c_idx]=1; u2[d_idx]=1
        v2 = [0]*(m*p); v2[e_idx]=1
        w2 = [0]*(p*n); w2[c10_idx]=1
        res.append((u2, v2, w2))
        
        # M3: a(f-h) -> C01 +
        u3 = [0]*(n*m); u3[a_idx]=1
        v3 = [0]*(m*p); v3[f_idx]=1; v3[h_idx]=-1
        w3 = [0]*(p*n); w3[c01_idx]=1
        res.append((u3, v3, w3))
        
        # M4: d(g-e) -> C10 +, C11 -
        u4 = [0]*(n*m); u4[d_idx]=1
        v4 = [0]*(m*p); v4[g_idx]=1; v4[e_idx]=-1
        w4 = [0]*(p*n); w4[c10_idx]=1; w4[c11_idx]=-1
        res.append((u4, v4, w4))
        
        # M5: (a+b)h -> C00 +, C01 +
        u5 = [0]*(n*m); u5[a_idx]=1; u5[b_idx]=1
        v5 = [0]*(m*p); v5[h_idx]=1
        w5 = [0]*(p*n); w5[c00_idx]=1; w5[c01_idx]=1
        res.append((u5, v5, w5))
        
        # M6: (c-a)(e+f) -> C11 +
        u6 = [0]*(n*m); u6[c_idx]=1; u6[a_idx]=-1
        v6 = [0]*(m*p); v6[e_idx]=1; v6[f_idx]=1
        w6 = [0]*(p*n); w6[c11_idx]=1
        res.append((u6, v6, w6))
        
        # M7: (b-d)(g+h) -> C00 -
        u7 = [0]*(n*m); u7[b_idx]=1; u7[d_idx]=-1
        v7 = [0]*(m*p); v7[g_idx]=1; v7[h_idx]=1
        w7 = [0]*(p*n); w7[c00_idx]=-1
        res.append((u7, v7, w7))
        
        return res

    # A11 B11: 2x2x6
    # A11: i in 0,1; j in 0,1
    # B11: j in 0,1; k in 0..5
    # Split k into 3 blocks of 2
    for r in range(3):
        k0 = 2*r
        k1 = 2*r + 1
        
        # A indices
        a_idx = 0*5 + 0 # i=0, j=0
        b_idx = 0*5 + 1 # i=0, j=1
        c_idx = 1*5 + 0 # i=1, j=0
        d_idx = 1*5 + 1 # i=1, j=1
        
        # B indices for block r
        e_idx = 0*6 + k0 # j=0, k=k0
        f_idx = 0*6 + k1 # j=0, k=k1
        g_idx = 1*6 + k0 # j=1, k=k0
        h_idx = 1*6 + k1 # j=1, k=k1
        
        # C indices for block r
        # C is 2x6. i in 0,1. k in k0, k1.
        c00_idx = k0*3 + 0
        c01_idx = k1*3 + 0
        c10_idx = k0*3 + 1
        c11_idx = k1*3 + 1
        
        terms.extend(get_strassen_terms([a_idx, b_idx, c_idx, d_idx], [e_idx, f_idx, g_idx, h_idx], [c00_idx, c01_idx, c10_idx, c11_idx]))
        
    # A12 B12: 2x3x6
    # A12: i in 0,1; j in 2,3,4
    # B12: j in 2,3,4; k in 0..5
    # Split k into 3 blocks of 2
    for r in range(3):
        k0 = 2*r
        k1 = 2*r + 1
        
        # Split A12 into A_L (j=2,3) and A_R (j=4)
        # Split B12 into B_T (j=2,3) and B_B (j=4)
        
        # A_L B_T: 2x2x2
        # A_L: i in 0,1; j in 2,3
        a_idx = 0*5 + 2
        b_idx = 0*5 + 3
        c_idx = 1*5 + 2
        d_idx = 1*5 + 3
        
        # B_T: j in 2,3; k in k0, k1
        e_idx = 2*6 + k0
        f_idx = 2*6 + k1
        g_idx = 3*6 + k0
        h_idx = 3*6 + k1
        
        # C_L: i in 0,1; k in k0, k1
        c00_idx = k0*3 + 0
        c01_idx = k1*3 + 0
        c10_idx = k0*3 + 1
        c11_idx = k1*3 + 1
        
        terms.extend(get_strassen_terms([a_idx, b_idx, c_idx, d_idx], [e_idx, f_idx, g_idx, h_idx], [c00_idx, c01_idx, c10_idx, c11_idx]))
        
        # A_R B_B: 2x1x2
        # A_R: i in 0,1; j=4
        # B_B: j=4; k in k0, k1
        # C_R: i in 0,1; k in k0, k1
        
        # Terms:
        # u1 v1 -> C00
        # u1 v2 -> C01
        # u2 v1 -> C10
        # u2 v2 -> C11
        
        u1_idx = 0*5 + 4
        u2_idx = 1*5 + 4
        v1_idx = 4*6 + k0
        v2_idx = 4*6 + k1
        
        # Term 1: u1 v1 -> C00
        u_vec = [0]*(n*m); u_vec[u1_idx]=1
        v_vec = [0]*(m*p); v_vec[v1_idx]=1
        w_vec = [0]*(p*n); w_vec[c00_idx]=1
        terms.append((u_vec, v_vec, w_vec))
        
        # Term 2: u1 v2 -> C01
        u_vec = [0]*(n*m); u_vec[u1_idx]=1
        v_vec = [0]*(m*p); v_vec[v2_idx]=1
        w_vec = [0]*(p*n); w_vec[c01_idx]=1
        terms.append((u_vec, v_vec, w_vec))
        
        # Term 3: u2 v1 -> C10
        u_vec = [0]*(n*m); u_vec[u2_idx]=1
        v_vec = [0]*(m*p); v_vec[v1_idx]=1
        w_vec = [0]*(p*n); w_vec[c10_idx]=1
        terms.append((u_vec, v_vec, w_vec))
        
        # Term 4: u2 v2 -> C11
        u_vec = [0]*(n*m); u_vec[u2_idx]=1
        v_vec = [0]*(m*p); v_vec[v2_idx]=1
        w_vec = [0]*(p*n); w_vec[c11_idx]=1
        terms.append((u_vec, v_vec, w_vec))
        
    # Populate U, V, W
    for c_idx, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][c_idx] = u_vec[i]
        for j in range(len(v_vec)):
            if v_vec[j] != 0:
                V[j][c_idx] = v_vec[j]
        for k in range(len(w_vec)):
            if w_vec[k] != 0:
                W[k][c_idx] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
