"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 48 decomposition for 2x5x5 matrix multiplication.
    # Uses Strassen (rank 7) for two 2x2x2 blocks and schoolbook for the rest.
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices.
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Start with empty lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Helper to create basis vectors
    def e(n, i):
        v = [0] * n
        v[i] = 1
        return v
        
    # Block 1: j in {0,1}, k in {0,1} (indices 0-based)
    # i in {0,1}
    # Strassen decomposition for 2x2x2
    # Indices for A: i*m + j. m=5.
    # Indices for B: j*p + k. p=5.
    # Indices for C: k*n + i. n=2.
    
    # Strassen terms for 2x2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = AB
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M3
    # C01 = M4 + M5
    # C10 = M6 + M7
    # C11 = M1 - M5 + M7
    
    # We need to map these to our global indices.
    # Block 1: j=0,1 and k=0,1.
    # A indices: i*5 + j.
    # B indices: j*5 + k.
    # C indices: k*2 + i.
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = [0]*10
    u1[0*5+0] = 1 # A00
    u1[1*5+1] = 1 # A11
    v1 = [0]*25
    v1[0*5+0] = 1 # B00
    v1[1*5+1] = 1 # B11
    w1 = [0]*10
    # C00 gets +1, C11 gets +1
    w1[0*2+0] = 1 # C00
    w1[1*2+1] = 1 # C11
    add_term(u1, v1, w1)
    
    # Term 2: (A10+A11)B00
    u2 = [0]*10
    u2[1*5+0] = 1 # A10
    u2[1*5+1] = 1 # A11
    v2 = [0]*25
    v2[0*5+0] = 1 # B00
    w2 = [0]*10
    # C00 gets -1, C10 gets +1
    w2[0*2+0] = -1 # C00
    w2[0*2+1] = 1 # C10
    add_term(u2, v2, w2)
    
    # Term 3: A00(B10-B11)
    u3 = [0]*10
    u3[0*5+0] = 1 # A00
    v3 = [0]*25
    v3[1*5+0] = 1 # B10
    v3[1*5+1] = -1 # B11
    w3 = [0]*10
    # C00 gets +1, C10 gets +1
    w3[0*2+0] = 1 # C00
    w3[0*2+1] = 1 # C10
    add_term(u3, v3, w3)
    
    # Term 4: A11(B01-B00)
    u4 = [0]*10
    u4[1*5+1] = 1 # A11
    v4 = [0]*25
    v4[0*5+1] = 1 # B01
    v4[0*5+0] = -1 # B00
    w4 = [0]*10
    # C01 gets +1, C11 gets -1
    w4[1*2+0] = 1 # C01
    w4[1*2+1] = -1 # C11
    add_term(u4, v4, w4)
    
    # Term 5: (A00+A01)B11
    u5 = [0]*10
    u5[0*5+0] = 1 # A00
    u5[0*5+1] = 1 # A01
    v5 = [0]*25
    v5[1*5+1] = 1 # B11
    w5 = [0]*10
    # C01 gets +1, C11 gets -1
    w5[1*2+0] = 1 # C01
    w5[1*2+1] = -1 # C11
    add_term(u5, v5, w5)
    
    # Term 6: (A10-A00)(B00+B01)
    u6 = [0]*10
    u6[1*5+0] = 1 # A10
    u6[0*5+0] = -1 # A00
    v6 = [0]*25
    v6[0*5+0] = 1 # B00
    v6[0*5+1] = 1 # B01
    w6 = [0]*10
    # C10 gets +1, C11 gets +1
    w6[0*2+1] = 1 # C10
    w6[1*2+1] = 1 # C11
    add_term(u6, v6, w6)
    
    # Term 7: (A01-A11)(B10+B11)
    u7 = [0]*10
    u7[0*5+1] = 1 # A01
    u7[1*5+1] = -1 # A11
    v7 = [0]*25
    v7[1*5+0] = 1 # B10
    v7[1*5+1] = 1 # B11
    w7 = [0]*10
    # C10 gets +1, C11 gets +1
    w7[0*2+1] = 1 # C10
    w7[1*2+1] = 1 # C11
    add_term(u7, v7, w7)
    
    # Block 2: j in {2,3}, k in {2,3}
    # Similar Strassen decomposition, shifted indices.
    # j' = j - 2, k' = k - 2.
    # A index: i*5 + (j'+2)
    # B index: (j'+2)*5 + (k'+2)
    # C index: (k'+2)*2 + i
    
    # Term 8: (A00'+A11')(B00'+B11')
    u8 = [0]*10
    u8[0*5+2] = 1 # A02
    u8[1*5+3] = 1 # A13
    v8 = [0]*25
    v8[2*5+2] = 1 # B22
    v8[3*5+3] = 1 # B33
    w8 = [0]*10
    w8[2*2+0] = 1 # C20
    w8[3*2+1] = 1 # C31
    add_term(u8, v8, w8)
    
    # Term 9: (A10'+A11')B00'
    u9 = [0]*10
    u9[1*5+2] = 1 # A12
    u9[1*5+3] = 1 # A13
    v9 = [0]*25
    v9[2*5+2] = 1 # B22
    w9 = [0]*10
    w9[2*2+0] = -1 # C20
    w9[2*2+1] = 1 # C21
    add_term(u9, v9, w9)
    
    # Term 10: A00'(B10'-B11')
    u10 = [0]*10
    u10[0*5+2] = 1 # A02
    v10 = [0]*25
    v10[3*5+2] = 1 # B32
    v10[3*5+3] = -1 # B33
    w10 = [0]*10
    w10[2*2+0] = 1 # C20
    w10[2*2+1] = 1 # C21
    add_term(u10, v10, w10)
    
    # Term 11: A11'(B01'-B00')
    u11 = [0]*10
    u11[1*5+3] = 1 # A13
    v11 = [0]*25
    v11[2*5+3] = 1 # B23
    v11[2*5+2] = -1 # B22
    w11 = [0]*10
    w11[3*2+0] = 1 # C30
    w11[3*2+1] = -1 # C31
    add_term(u11, v11, w11)
    
    # Term 12: (A00'+A01')B11'
    u12 = [0]*10
    u12[0*5+2] = 1 # A02
    u12[0*5+3] = 1 # A03
    v12 = [0]*25
    v12[3*5+3] = 1 # B33
    w12 = [0]*10
    w12[3*2+0] = 1 # C30
    w12[3*2+1] = -1 # C31
    add_term(u12, v12, w12)
    
    # Term 13: (A10'-A00')(B00'+B01')
    u13 = [0]*10
    u13[1*5+2] = 1 # A12
    u13[0*5+2] = -1 # A02
    v13 = [0]*25
    v13[2*5+2] = 1 # B22
    v13[2*5+3] = 1 # B23
    w13 = [0]*10
    w13[2*2+1] = 1 # C21
    w13[3*2+1] = 1 # C31
    add_term(u13, v13, w13)
    
    # Term 14: (A01'-A11')(B10'+B11')
    u14 = [0]*10
    u14[0*5+3] = 1 # A03
    u14[1*5+3] = -1 # A13
    v14 = [0]*25
    v14[3*5+2] = 1 # B32
    v14[3*5+3] = 1 # B33
    w14 = [0]*10
    w14[2*2+1] = 1 # C21
    w14[3*2+1] = 1 # C31
    add_term(u14, v14, w14)
    
    # Remaining Schoolbook Terms
    # Set 3: j in {0,1}, k in {2,3}
    for j in [0, 1]:
        for k in [2, 3]:
            for i in [0, 1]:
                u = [0]*10
                u[i*5+j] = 1
                v = [0]*25
                v[j*5+k] = 1
                w = [0]*10
                w[k*2+i] = 1
                add_term(u, v, w)
                
    # Set 4: j in {2,3}, k in {0,1}
    for j in [2, 3]:
        for k in [0, 1]:
            for i in [0, 1]:
                u = [0]*10
                u[i*5+j] = 1
                v = [0]*25
                v[j*5+k] = 1
                w = [0]*10
                w[k*2+i] = 1
                add_term(u, v, w)
                
    # Set 5: j in {0,1,2,3}, k = 4
    for j in [0, 1, 2, 3]:
        for i in [0, 1]:
            u = [0]*10
            u[i*5+j] = 1
            v = [0]*25
            v[j*5+4] = 1
            w = [0]*10
            w[4*2+i] = 1
            add_term(u, v, w)
            
    # Set 6: j = 4, k in {0,1,2,3,4}
    for k in [0, 1, 2, 3, 4]:
        for i in [0, 1]:
            u = [0]*10
            u[i*5+4] = 1
            v = [0]*25
            v[4*5+k] = 1
            w = [0]*10
            w[k*2+i] = 1
            add_term(u, v, w)
            
    # Convert lists to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
