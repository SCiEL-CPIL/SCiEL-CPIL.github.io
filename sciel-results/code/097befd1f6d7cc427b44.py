def construct():
    n, m, p = 4, 5, 5
    
    def build_schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0]*(n*m); u[i*m+j] = 1
                    v = [0]*(m*p); v[j*p+k] = 1
                    w = [0]*(p*n); w[k*n+i] = 1
                    terms.append((u, v, w))
        return terms
    
    def build_with_strassen():
        terms = []
        
        def add_term(u, v, w, coeff=1.0):
            terms.append((coeff * u, coeff * v, coeff * w))
        
        def schoolbook(i_off, j_off, k_off, i_size, j_size, k_size):
            for li in range(i_size):
                for lj in range(j_size):
                    for lk in range(k_size):
                        i = i_off + li
                        j = j_off + lj
                        k = k_off + lk
                        u = [0]*(n*m); u[i*m+j] = 1
                        v = [0]*(m*p); v[j*p+k] = 1
                        w = [0]*(p*n); w[k*n+i] = 1
                        add_term(u, v, w, 1)
        
        def strassen_2x2x2(i_off, j_off, k_off):
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # p1 = (a00+a11)(b00+b11)
            u = [0]*(n*m); u[i0*m+j0] = 1; u[i1*m+j1] = 1
            v = [0]*(m*p); v[j0*p+k0] = 1; v[j1*p+k1] = 1
            w = [0]*(p*n); w[k0*n+i0] = 1; w[k1*n+i1] = 1
            add_term(u, v, w, 1)
            
            # p2 = (a10+a11)b00
            u = [0]*(n*m); u[i1*m+j0] = 1; u[i1*m+j1] = 1
            v = [0]*(m*p); v[j0*p+k0] = 1
            w = [0]*(p*n); w[k0*n+i1] = 1
            add_term(u, v, w, 1)
            
            # p3 = a00(b10-b11)
            u = [0]*(n*m); u[i0*m+j0] = 1
            v = [0]*(m*p); v[j1*p+k0] = 1; v[j1*p+k1] = -1
            w = [0]*(p*n); w[k1*n+i0] = 1
            add_term(u, v, w, 1)
            
            # p4 = a11(b00-b01)
            u = [0]*(n*m); u[i1*m+j1] = 1
            v = [0]*(m*p); v[j0*p+k0] = 1; v[j0*p+k1] = -1
            w = [0]*(p*n); w[k0*n+i0] = -1; w[k1*n+i0] = 1
            add_term(u, v, w, 1)
            
            # p5 = (a00+a01)b11
            u = [0]*(n*m); u[i0*m+j0] = 1; u[i0*m+j1] = 1
            v = [0]*(m*p); v[j1*p+k1] = 1
            w = [0]*(p*n); w[k1*n+i0] = 1; w[k1*n+i1] = 1
            add_term(u, v, w, 1)
            
            # p6 = (a10-a00)(b00+b01)
            u = [0]*(n*m); u[i1*m+j0] = 1; u[i0*m+j0] = -1
            v = [0]*(m*p); v[j0*p+k0] = 1; v[j0*p+k1] = 1
            w = [0]*(p*n); w[k1*n+i1] = 1
            add_term(u, v, w, 1)
            
            # p7 = (a01-a11)(b10+b11)
            u = [0]*(n*m); u[i0*m+j1] = 1; u[i1*m+j1] = -1
            v = [0]*(m*p); v[j1*p+k0] = 1; v[j1*p+k1] = 1
            w = [0]*(p*n); w[k0*n+i0] = 1; w[k1*n+i1] = 1
            add_term(u, v, w, 1)
        
        strassen_2x2x2(0, 0, 0)
        schoolbook(0, 2, 0, 2, 3, 2)
        schoolbook(0, 0, 2, 2, 2, 3)
        schoolbook(0, 2, 2, 2, 3, 3)
        strassen_2x2x2(2, 0, 0)
        schoolbook(2, 2, 0, 2, 3, 2)
        schoolbook(2, 0, 2, 2, 2, 3)
        schoolbook(2, 2, 2, 2, 3, 3)
        
        return terms
    
    # Verify Strassen decomposition
    terms_strassen = build_with_strassen()
    
    # Check if the decomposition is correct
    # For each (i,j,k), verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    rank = len(terms_strassen)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (u, v, w) in enumerate(terms_strassen):
        for idx in range(n*m):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(m*p):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(p*n):
            if w[idx] != 0:
                W[idx][r] = w[idx]
    
    # Verify
    correct = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-10:
                    correct = False
                    break
            if not correct:
                break
        if not correct:
            break
    
    if not correct:
        # Fall back to schoolbook
        terms = build_schoolbook()
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r, (u, v, w) in enumerate(terms):
            for idx in range(n*m):
                if u[idx] != 0:
                    U[idx][r] = u[idx]
            for idx in range(m*p):
                if v[idx] != 0:
                    V[idx][r] = v[idx]
            for idx in range(p*n):
                if w[idx] != 0:
                    W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}
