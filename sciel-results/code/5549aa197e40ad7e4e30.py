DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    terms = []  # Each term: (U_list, V_list, W_list)
    
    def make_term(U_dict, V_dict, W_dict):
        U = [U_dict.get(i*m+j, 0) for i in range(n) for j in range(m)]
        V = [V_dict.get(j*p+k, 0) for j in range(m) for k in range(p)]
        W = [W_dict.get(k*n+i, 0) for k in range(p) for i in range(n)]
        terms.append((U, V, W))
    
    # Part 1: 2x2x10 part (j=0,1), Strassen on column pairs
    for pair_idx in range(p // 2):
        k0 = 2 * pair_idx
        k1 = k0 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        make_term({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1})
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        make_term({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1})
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        make_term({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1})
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        make_term({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1})
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        make_term({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1})
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        make_term({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1})
        # M7 = (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k0])
        make_term({(0,1):1, (1,1):-1}, {(1,k1):1, (1,k0):1}, {(k0,0):1})
    
    # Part 2: 2x1x10 part (j=2)
    for i in range(n):
        for k in range(p):
            make_term({(i,2):1}, {(2,k):1}, {(k,i):1})
    
    rank = len(terms)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r, (U_vec, V_vec, W_vec) in enumerate(terms):
        for idx in range(n*m):
            U_mat[idx][r] = U_vec[idx]
        for idx in range(m*p):
            V_mat[idx][r] = V_vec[idx]
        for idx in range(p*n):
            W_mat[idx][r] = W_vec[idx]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
