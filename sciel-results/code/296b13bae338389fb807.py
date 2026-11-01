import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # U: (4, R), V: (22, R), W: (22, R)
    
    # We use 5 blocks of 2x2 (Strassen, rank 7) and 1 block of 2x1 (Trivial, rank 4)
    # Total rank = 5*7 + 4 = 39
    R = 39
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1
        
    # Process 5 blocks of 2x2
    for b in range(5):
        # Columns 2b and 2b+1
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Indices in V (m*p = 22): row j, col k -> j*11 + k
        # v_00 = B[0, c0], v_01 = B[0, c1]
        # v_10 = B[1, c0], v_11 = B[1, c1]
        
        # Indices in W (p*n = 22): col k, row i -> k*2 + i
        # w_00 = C[0, c0], w_01 = C[0, c1]
        # w_10 = C[1, c0], w_11 = C[1, c1]
        
        # Strassen Terms
        # 1. (a+d)(e+h)
        # U: A[0,0]+A[1,1] -> [1,0,0,1]
        # V: B[0,c0]+B[1,c1]
        # W: C[0,c0]+C[1,c1]
        u1 = np.zeros(4); u1[0]=1; u1[3]=1
        v1 = np.zeros(22); v1[0*11+c0]=1; v1[1*11+c1]=1
        w1 = np.zeros(22); w1[c0*2+0]=1; w1[c1*2+1]=1
        add_term(u1, v1, w1)
        
        # 2. (c+d)e
        # U: A[1,0]+A[1,1] -> [0,0,1,1]
        # V: B[0,c0]
        # W: C[1,c0]-C[1,c1]
        u2 = np.zeros(4); u2[2]=1; u2[3]=1
        v2 = np.zeros(22); v2[0*11+c0]=1
        w2 = np.zeros(22); w2[c0*2+1]=1; w2[c1*2+1]=-1
        add_term(u2, v2, w2)
        
        # 3. a(f-h)
        # U: A[0,0] -> [1,0,0,0]
        # V: B[0,c1]-B[1,c0]
        # W: C[0,c1]+C[1,c1]
        u3 = np.zeros(4); u3[0]=1
        v3 = np.zeros(22); v3[0*11+c1]=1; v3[1*11+c0]=-1
        w3 = np.zeros(22); w3[c1*2+0]=1; w3[c1*2+1]=1
        add_term(u3, v3, w3)
        
        # 4. d(g-e)
        # U: A[1,1] -> [0,0,0,1]
        # V: B[1,c0]-B[0,c0]
        # W: C[0,c0]+C[1,c0]
        u4 = np.zeros(4); u4[3]=1
        v4 = np.zeros(22); v4[1*11+c0]=1; v4[0*11+c0]=-1
        w4 = np.zeros(22); w4[c0*2+0]=1; w4[c0*2+1]=1
        add_term(u4, v4, w4)
        
        # 5. (a+b)h
        # U: A[0,0]+A[0,1] -> [1,1,0,0]
        # V: B[1,c1]
        # W: -C[0,c0]+C[0,c1]
        u5 = np.zeros(4); u5[0]=1; u5[1]=1
        v5 = np.zeros(22); v5[1*11+c1]=1
        w5 = np.zeros(22); w5[c0*2+0]=-1; w5[c1*2+0]=1
        add_term(u5, v5, w5)
        
        # 6. (c-a)(e+f)
        # U: A[1,0]-A[0,0] -> [-1,0,1,0]
        # V: B[0,c0]+B[0,c1]
        # W: C[1,c1]
        u6 = np.zeros(4); u6[0]=-1; u6[2]=1
        v6 = np.zeros(22); v6[0*11+c0]=1; v6[0*11+c1]=1
        w6 = np.zeros(22); w6[c1*2+1]=1
        add_term(u6, v6, w6)
        
        # 7. (b-d)(g+h)
        # U: A[0,1]-A[1,1] -> [0,1,0,-1]
        # V: B[1,c0]+B[1,c1]
        # W: C[0,c0]
        u7 = np.zeros(4); u7[1]=1; u7[3]=-1
        v7 = np.zeros(22); v7[1*11+c0]=1; v7[1*11+c1]=1
        w7 = np.zeros(22); w7[c0*2+0]=1
        add_term(u7, v7, w7)
        
    # Process last block of 2x1 (column 10)
    c0 = 10
    
    # 1. A[0,0]B[0,10]C[0,10]
    u = np.zeros(4); u[0]=1
    v = np.zeros(22); v[0*11+c0]=1
    w = np.zeros(22); w[c0*2+0]=1
    add_term(u, v, w)
    
    # 2. A[0,1]B[1,10]C[0,10]
    u = np.zeros(4); u[1]=1
    v = np.zeros(22); v[1*11+c0]=1
    w = np.zeros(22); w[c0*2+0]=1
    add_term(u, v, w)
    
    # 3. A[1,0]B[0,10]C[1,10]
    u = np.zeros(4); u[2]=1
    v = np.zeros(22); v[0*11+c0]=1
    w = np.zeros(22); w[c0*2+1]=1
    add_term(u, v, w)
    
    # 4. A[1,1]B[1,10]C[1,10]
    u = np.zeros(4); u[3]=1
    v = np.zeros(22); v[1*11+c0]=1
    w = np.zeros(22); w[c0*2+1]=1
    add_term(u, v, w)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
