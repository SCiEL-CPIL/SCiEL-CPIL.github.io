def construct():
    n, m, p = 3, 4, 7
    # U: (12, rank), V: (28, rank), W: (21, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def add_strassen(i1, i2, j1, j2, k1, k2):
        # M1 = (a11+a22)(b11+b22)
        u = {i1*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k1*n+i2: 1, k2*n+i1: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M2 = (a21+a22)b11
        u = {i2*m+j1: 1, i2*m+j2: 1}
        v = {j1*p+k1: 1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M3 = a11(b12-b22)
        u = {i1*m+j1: 1}
        v = {j1*p+k2: 1, j2*p+k2: -1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M4 = a22(b21-b11)
        u = {i2*m+j2: 1}
        v = {j2*p+k1: 1, j1*p+k1: -1}
        w = {k1*n+i2: 1}
        add_term(u, v, w)
        
        # M5 = (a11+a12)b22
        u = {i1*m+j1: 1, i1*m+j2: 1}
        v = {j2*p+k2: 1}
        w = {k2*n+i1: 1}
        add_term(u, v, w)
        
        # M6 = (a21-a11)(b11+b12)
        u = {i2*m+j1: 1, i1*m+j1: -1}
        v = {j1*p+k1: 1, j1*p+k2: 1}
        w = {k1*n+i2: 1, k2*n+i2: 1}
        add_term(u, v, w)
        
        # M7 = (a12-a22)(b21+b22)
        u = {i1*m+j2: 1, i2*m+j2: -1}
        v = {j2*p+k1: 1, j2*p+k2: 1}
        w = {k1*n+i1: 1, k2*n+i1: 1}
        add_term(u, v, w)
        
    def add_2x2x1(i1, i2, j1, j2, k):
        # c11 = a11*b1 + a12*b2
        # c12 = a21*b1 + a22*b2
        # This is 4 rank-1 terms?
        # Actually, 2x2x1 is rank 4.
        # T = a11 b1 w11 + a12 b2 w11 + a21 b1 w21 + a22 b2 w21
        # Wait, W is (p*n, rank).
        # W index is k*n + i.
        
        # Term 1: a11 b1 -> w11
        u = {i1*m+j1: 1}
        v = {j1*p+k: 1}
        w = {k*n+i1: 1}
        add_term(u, v, w)
        
        # Term 2: a12 b2 -> w11
        u = {i1*m+j2: 1}
        v = {j2*p+k: 1}
        w = {k*n+i1: 1}
        add_term(u, v, w)
        
        # Term 3: a21 b1 -> w21
        u = {i2*m+j1: 1}
        v = {j1*p+k: 1}
        w = {k*n+i2: 1}
        add_term(u, v, w)
        
        # Term 4: a22 b2 -> w21
        u = {i2*m+j2: 1}
        v = {j2*p+k: 1}
        w = {k*n+i2: 1}
        add_term(u, v, w)

    # Block 1: I=[0,1]
    # J1=[0,1], K splits [0,1], [2,3], [4,5], [6]
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 4, 5)
    add_2x2x1(0, 1, 0, 1, 6)
    
    # J2=[2,3], K splits [0,1], [2,3], [4,5], [6]
    add_strassen(0, 1, 2, 3, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3)
    add_strassen(0, 1, 2, 3, 4, 5)
    add_2x2x1(0, 1, 2, 3, 6)
    
    # Block 2: I=[2]
    # J=[0,3], K=[0,6]
    for j in range(4):
        for k in range(7):
            u = {2*m+j: 1}
            v = {j*p+k: 1}
            w = {k*n+2: 1}
            add_term(u, v, w)
            
    rank = len(U_terms)
    
    # Convert dicts to lists
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for idx, (u, v, w) in enumerate(zip(U_terms, V_terms, W_terms)):
        for key, val in u.items():
            U[key][idx] = val
        for key, val in v.items():
            V[key][idx] = val
        for key, val in w.items():
            W[key][idx] = val
            
    return {"U": U, "V": V, "W": W}
