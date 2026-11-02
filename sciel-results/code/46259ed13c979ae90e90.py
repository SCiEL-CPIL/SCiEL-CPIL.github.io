import numpy as np

def construct():
    n, m, p = 2, 3, 8
    nm, mp, pn = n*m, m*p, p*n  # 6, 24, 16
    
    rank = 44
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Indices for U (A: 2x3)
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3 (A_22)
    # A[0,2]=4, A[1,2]=5 (A_21)
    
    # Indices for V (B: 3x8)
    # B[0,0..7] = 0..7
    # B[1,0..7] = 8..15
    # B[2,0..7] = 16..23
    # B_22 blocks (2x2) from B rows 0,1
    # Block 0: B[0,0]=0, B[0,1]=1, B[1,0]=8, B[1,1]=9
    # Block 1: B[0,2]=2, B[0,3]=3, B[1,2]=10, B[1,3]=11
    # Block 2: B[0,4]=4, B[0,5]=5, B[1,4]=12, B[1,5]=13
    # Block 3: B[0,6]=6, B[0,7]=7, B[1,6]=14, B[1,7]=15
    # B_18: B[2,0..7] = 16..23
    
    # Indices for W (C: 2x8)
    # C[0,0..7] = 0..7
    # C[1,0..7] = 8..15
    
    t = 0
    
    # Helper to add a term
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        nonlocal U, V, W, t
        U[u_idx, t] += u_coeff
        V[v_idx, t] += v_coeff
        W[w_idx, t] += w_coeff
        t += 1

    # Strassen helper for 2x2
    # a_idx, b_idx, c_idx, d_idx for A
    # e_idx, f_idx, g_idx, h_idx for B
    # c11_idx, c12_idx, c21_idx, c22_idx for C
    def add_strassen(a, b, c, d, e, f, g, h, c11, c12, c21, c22):
        nonlocal U, V, W, t
        
        # M1 = (a+d)(e+h)
        U[a, t] += 1; U[d, t] += 1
        V[e, t] += 1; V[h, t] += 1
        W[c11, t] += 1; W[c22, t] += 1
        t += 1
        
        # M2 = (c+d)e
        U[c, t] += 1; U[d, t] += 1
        V[e, t] += 1
        W[c21, t] += 1; W[c22, t] -= 1
        t += 1
        
        # M3 = a(f-h)
        U[a, t] += 1
        V[f, t] += 1; V[h, t] -= 1
        W[c12, t] += 1; W[c22, t] += 1
        t += 1
        
        # M4 = d(g-e)
        U[d, t] += 1
        V[g, t] += 1; V[e, t] -= 1
        W[c11, t] += 1; W[c21, t] += 1
        t += 1
        
        # M5 = (a+b)h
        U[a, t] += 1; U[b, t] += 1
        V[h, t] += 1
        W[c11, t] -= 1; W[c12, t] += 1
        t += 1
        
        # M6 = (c-a)(e+f)
        U[c, t] += 1; U[a, t] -= 1
        V[e, t] += 1; V[f, t] += 1
        W[c22, t] += 1
        t += 1
        
        # M7 = (b-d)(g+h)
        U[b, t] += 1; U[d, t] -= 1
        V[g, t] += 1; V[h, t] += 1
        W[c11, t] += 1
        t += 1

    # Block 0: Cols 0,1
    # A_22: a=0, b=1, c=2, d=3
    # B_22: e=0, f=1, g=8, h=9
    # C: c11=0, c12=1, c21=8, c22=9
    add_strassen(0, 1, 2, 3, 0, 1, 8, 9, 0, 1, 8, 9)
    
    # Block 1: Cols 2,3
    # B_22: e=2, f=3, g=10, h=11
    # C: c11=2, c12=3, c21=10, c22=11
    add_strassen(0, 1, 2, 3, 2, 3, 10, 11, 2, 3, 10, 11)
    
    # Block 2: Cols 4,5
    # B_22: e=4, f=5, g=12, h=13
    # C: c11=4, c12=5, c21=12, c22=13
    add_strassen(0, 1, 2, 3, 4, 5, 12, 13, 4, 5, 12, 13)
    
    # Block 3: Cols 6,7
    # B_22: e=6, f=7, g=14, h=15
    # C: c11=6, c12=7, c21=14, c22=15
    add_strassen(0, 1, 2, 3, 6, 7, 14, 15, 6, 7, 14, 15)
    
    # Standard part: A_21 (2x1) * B_18 (1x8)
    # A_21: a=4 (row 0), c=5 (row 1)
    # B_18: e_j = 16+j
    # C_1j = j, C_2j = 8+j
    
    for j in range(8):
        # C_1j = A_21[0] * B_18[j]
        add_term(4, 1, 16+j, 1, j, 1)
        # C_2j = A_21[1] * B_18[j]
        add_term(5, 1, 16+j, 1, 8+j, 1)
        
    # Verification (optional, but good for debug)
    # U, V, W are built.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
