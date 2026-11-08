def construct():
    n, m, p = 3, 3, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Part 1: A[:,2] * B[2,:] (outer product, rank 3)
    # A[i,2] * B[2,k] for i=0,1,2; k=0,1,2,3,4
    # U index for A[i,2]: i*3+2
    # V index for B[2,k]: 2*5+k
    # W index for C[i,k]: k*3+i
    for i in range(3):
        u = [0]*9
        v = [0]*15
        w = [0]*15
        u[i*3+2] = 1
        for k in range(5):
            v[2*5+k] = 1
            w[k*3+i] = 1
        add_term(u, v, w)
    
    # Part 2: A[2,:2] * B[:2,:] (1x2 * 2x5, rank 2)
    # A[2,j] * B[j,k] for j=0,1; k=0,1,2,3,4
    # U index for A[2,j]: 2*3+j
    # V index for B[j,k]: j*5+k
    # W index for C[2,k]: k*3+2
    for j in range(2):
        u = [0]*9
        v = [0]*15
        w = [0]*15
        u[2*3+j] = 1
        for k in range(5):
            v[j*5+k] = 1
            w[k*3+2] = 1
        add_term(u, v, w)
    
    # Part 3: A[:2,:2] * B[:2,:2] (2x2 * 2x2, Strassen rank 7)
    # A[i,j] for i=0,1; j=0,1: U index i*3+j
    # B[j,k] for j=0,1; k=0,1: V index j*5+k
    # C[i,k] for i=0,1; k=0,1: W index k*3+i
    
    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,1,-1], [0,1,0,0]),
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,-1,1,0], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    a_map = [0,1,3,4]  # A[0,0],A[0,1],A[1,0],A[1,1]
    b_map = [0,1,5,6]  # B[0,0],B[0,1],B[1,0],B[1,1]
    c_map = [0,3,1,4]  # C[0,0],C[0,1],C[1,0],C[1,1]
    
    for u_l, v_l, w_l in strassen:
        u = [0]*9; v = [0]*15; w = [0]*15
        for idx, val in enumerate(u_l):
            if val: u[a_map[idx]] = val
        for idx, val in enumerate(v_l):
            if val: v[b_map[idx]] = val
        for idx, val in enumerate(w_l):
            if val: w[c_map[idx]] = val
        add_term(u, v, w)
    
    # Part 4: A[:2,:2] * B[:2,2:4] (2x2 * 2x2, Strassen rank 7)
    # B[j,k] for j=0,1; k=2,3: V index j*5+k
    # C[i,k] for i=0,1; k=2,3: W index k*3+i
    b_map2 = [2,3,7,8]  # B[0,2],B[0,3],B[1,2],B[1,3]
    c_map2 = [6,9,7,10]  # C[0,2],C[0,3],C[1,2],C[1,3]
    
    for u_l, v_l, w_l in strassen:
        u = [0]*9; v = [0]*15; w = [0]*15
        for idx, val in enumerate(u_l):
            if val: u[a_map[idx]] = val
        for idx, val in enumerate(v_l):
            if val: v[b_map2[idx]] = val
        for idx, val in enumerate(w_l):
            if val: w[c_map2[idx]] = val
        add_term(u, v, w)
    
    # Part 5: A[:2,:2] * B[:2,4] (2x2 * 2x1, rank 4)
    # A[i,j]*B[j,4] for i=0,1; j=0,1
    # C[i,4]: W index 4*3+i
    for i in range(2):
        for j in range(2):
            u = [0]*9; v = [0]*15; w = [0]*15
            u[i*3+j] = 1
            v[j*5+4] = 1
            w[4*3+i] = 1
            add_term(u, v, w)
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
