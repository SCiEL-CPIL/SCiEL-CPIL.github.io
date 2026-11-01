def construct():
    n, m, p = 2, 2, 13
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    def add_strassen(k1, k2):
        # M1
        add_term({0:1, 3:1}, {k1:1, 13+k2:1}, {k1:1, 2*k2+1:1})
        # M2
        add_term({2:1, 3:1}, {k1:1}, {2*k1+1:1})
        # M3
        add_term({0:1}, {13+k1:1, k1:-1}, {k2:1})
        # M4
        add_term({3:1}, {k2:1, 13+k1:-1}, {k1:1, 2*k1+1:1})
        # M5
        add_term({0:1, 1:1}, {13+k2:1}, {k1:-1, k2:1})
        # M6
        add_term({2:1, 0:-1}, {k2:1, 13+k2:1}, {2*k2+1:1}) # M6 = (A10-A00)(B01+B11)
        # M7
        add_term({1:1, 2:-1}, {k1:1, 13+k1:1}, {k1:1}) # M7 = (A01-A10)(B00+B10)
        
    def add_schoolbook(k):
        # C0k = A00 B0k + A01 B1k
        # C1k = A10 B0k + A11 B1k
        add_term({0:1}, {k:1}, {k:1})
        add_term({1:1}, {13+k:1}, {k:1})
        add_term({2:1}, {k:1}, {2*k+1:1})
        add_term({3:1}, {13+k:1}, {2*k+1:1})
        
    # 6 Strassen blocks for k=0..11
    for i in range(6):
        add_strassen(2*i, 2*i+1)
        
    # 1 Schoolbook block for k=12
    add_schoolbook(12)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(26)]
    W = [[0]*rank for _ in range(26)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
