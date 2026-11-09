import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (14, rank)
    
    # We will build the decomposition in parts.
    # Part 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # This is M_{2,2,7}
    # Part 2: A[0:2, 2:3] * B[2:3, :] -> C[0:2, :]
    # This is M_{2,1,7}
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a rank-1 tensor
    # u: vector of length n*m
    # v: vector of length m*p
    # w: vector of length p*n
    def add_tensor(u, v, w):
        U_parts.append(u)
        V_parts.append(v)
        W_parts.append(w)

    # --- Part 1: M_{2,2,7} ---
    # A indices: i in 0,1; j in 0,1. U index: i*m + j = i*3 + j.
    # B indices: j in 0,1; k in 0..6. V index: j*p + k = j*7 + k.
    # C indices: i in 0,1; k in 0..6. W index: k*n + i = k*2 + i.
    
    # Strassen for 2x2x2
    def strassen_2x2x2(k_offset):
        # k_offset is the starting column index in B (0, 2, 4)
        # B cols are k_offset, k_offset+1
        # A cols are 0, 1
        # C cols are k_offset, k_offset+1
        
        # m1 = (a+d)(e+h)
        # A: a=A00, d=A11. B: e=B(k_offset,0), h=B(k_offset+1,1) -- wait, B is j x k.
        # B[0, k_offset], B[1, k_offset+1]
        
        # Let's define the linear forms for U and V.
        # U vector has non-zeros at indices:
        # A00: 0*3+0=0
        # A11: 1*3+1=4
        # etc.
        
        # m1: (A00 + A11) * (B0k0 + B1k1)
        u1 = [0]*6
        u1[0] = 1 # A00
        u1[4] = 1 # A11
        
        v1 = [0]*21
        v1[0*7 + k_offset] = 1 # B0, k_offset
        v1[1*7 + (k_offset+1)] = 1 # B1, k_offset+1
        
        w1 = [0]*14
        # Contribution to C:
        # C00 += m1, C01 += m1, C10 += m1, C11 += m1
        # C indices: k*n + i
        # C00: k=k_offset, i=0 -> (k_offset)*2 + 0
        # C01: k=k_offset+1, i=0 -> (k_offset+1)*2 + 0
        # C10: k=k_offset, i=1 -> (k_offset)*2 + 1
        # C11: k=k_offset+1, i=1 -> (k_offset+1)*2 + 1
        w1[k_offset*2 + 0] = 1
        w1[(k_offset+1)*2 + 0] = 1
        w1[k_offset*2 + 1] = 1
        w1[(k_offset+1)*2 + 1] = 1
        add_tensor(u1, v1, w1)
        
        # m2 = (A10 + A11) * B0k0
        u2 = [0]*6
        u2[3] = 1 # A10
        u2[4] = 1 # A11
        v2 = [0]*21
        v2[0*7 + k_offset] = 1
        w2 = [0]*14
        w2[k_offset*2 + 1] = 1 # C10
        w2[(k_offset+1)*2 + 1] = -1 # C11
        add_tensor(u2, v2, w2)
        
        # m3 = A00 * (B1k0 - B0k0) -- wait, standard Strassen:
        # m3 = a(f-h) where f=B0k1, h=B1k1?
        # Let's stick to the standard definition:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # e=B0k0, f=B0k1, g=B1k0, h=B1k1
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+h)
        # m7 = (b-d)(e+g)
        
        # Correcting m2 above:
        # I used (c+d)e. Correct.
        
        # m3 = a(f-h)
        u3 = [0]*6
        u3[0] = 1 # A00
        v3 = [0]*21
        v3[0*7 + (k_offset+1)] = 1 # B0, k1
        v3[1*7 + (k_offset+1)] = -1 # B1, k1
        w3 = [0]*14
        w3[(k_offset+1)*2 + 0] = 1 # C01
        w3[(k_offset+1)*2 + 1] = 1 # C11
        add_tensor(u3, v3, w3)
        
        # m4 = d(g-e)
        u4 = [0]*6
        u4[4] = 1 # A11
        v4 = [0]*21
        v4[1*7 + k_offset] = 1 # B1, k0
        v4[0*7 + k_offset] = -1 # B0, k0
        w4 = [0]*14
        w4[k_offset*2 + 0] = 1 # C00
        w4[k_offset*2 + 1] = 1 # C10
        add_tensor(u4, v4, w4)
        
        # m5 = (a+b)h
        u5 = [0]*6
        u5[0] = 1 # A00
        u5[1] = 1 # A01
        v5 = [0]*21
        v5[1*7 + (k_offset+1)] = 1 # B1, k1
        w5 = [0]*14
        w5[k_offset*2 + 0] = -1 # C00
        w5[(k_offset+1)*2 + 0] = 1 # C01
        add_tensor(u5, v5, w5)
        
        # m6 = (c-d)(f+h)
        u6 = [0]*6
        u6[3] = 1 # A10
        u6[4] = -1 # A11
        v6 = [0]*21
        v6[0*7 + (k_offset+1)] = 1 # B0, k1
        v6[1*7 + (k_offset+1)] = 1 # B1, k1
        w6 = [0]*14
        w6[(k_offset+1)*2 + 1] = 1 # C11
        add_tensor(u6, v6, w6)
        
        # m7 = (b-d)(e+g)
        u7 = [0]*6
        u7[1] = 1 # A01
        u7[4] = -1 # A11
        v7 = [0]*21
        v7[0*7 + k_offset] = 1 # B0, k0
        v7[1*7 + k_offset] = 1 # B1, k0
        w7 = [0]*14
        w7[k_offset*2 + 0] = 1 # C00
        add_tensor(u7, v7, w7)

    # Apply Strassen to blocks
    strassen_2x2x2(0)
    strassen_2x2x2(2)
    strassen_2x2x2(4)
    
    # Block 4: col 6. M_{2,2,1}
    # C_i6 = A_i0 B_06 + A_i1 B_16
    # 4 terms:
    # A00 B06 -> C06
    u = [0]*6; u[0]=1
    v = [0]*21; v[0*7+6]=1
    w = [0]*14; w[6*2+0]=1
    add_tensor(u,v,w)
    
    # A01 B16 -> C06
    u = [0]*6; u[1]=1
    v = [0]*21; v[1*7+6]=1
    w = [0]*14; w[6*2+0]=1
    add_tensor(u,v,w)
    
    # A10 B06 -> C16
    u = [0]*6; u[3]=1
    v = [0]*21; v[0*7+6]=1
    w = [0]*14; w[6*2+1]=1
    add_tensor(u,v,w)
    
    # A11 B16 -> C16
    u = [0]*6; u[4]=1
    v = [0]*21; v[1*7+6]=1
    w = [0]*14; w[6*2+1]=1
    add_tensor(u,v,w)

    # --- Part 2: M_{2,1,7} ---
    # A col 2. B row 2.
    # A indices: i in 0,1; j=2. U index: i*3+2.
    # B indices: j=2; k in 0..6. V index: 2*7+k.
    # C indices: i in 0,1; k in 0..6. W index: k*2+i.
    
    for i in range(2):
        for k in range(7):
            u = [0]*6
            u[i*3+2] = 1
            v = [0]*21
            v[2*7+k] = 1
            w = [0]*14
            w[k*2+i] = 1
            add_tensor(u,v,w)

    # Convert to numpy arrays
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {"U": U, "V": V, "W": W}
