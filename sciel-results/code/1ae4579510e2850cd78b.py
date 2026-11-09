def construct():
    n, m, p = DIMENSIONS
    import time
    deadline = time.time() + 110
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Use Strassen for 2x2x2 blocks
    # Partition: i in {0,1}, {2}; j in {0,1}, {2,3}, {4}; k in {0,1}, {2,3}, {4,5}
    
    # Strassen 2x2x2 terms (local indices 0,1 for i,j,k):
    # Each term is a triple (U_local, V_local, W_local) where each is a 4-element vector
    
    strassen_terms = [
        # Term 1: m1 = (a00+a11)(b00+b11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # Term 2: m2 = (a10+a11)b00
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # Term 3: m3 = a00(b10-b11)
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        # Term 4: m4 = a11(b01-b00)
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
        # Term 5: m5 = (a00+a01)b11
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # Term 6: m6 = (a10-a00)(b00+b10)
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
        # Term 7: m7 = (a01-a11)(b01+b11)
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    # Verify Strassen locally
    def verify_strassen_local():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for U_loc, V_loc, W_loc in strassen_terms:
                        s += U_loc[i*2+j] * V_loc[j*2+k] * W_loc[k*2+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    assert verify_strassen_local(), "Strassen local verification failed"
    
    # Now build the global decomposition
    # Partition indices
    i_groups = [(0,1), (2,)]
    j_groups = [(0,1), (2,3), (4,)]
    k_groups = [(0,1), (2,3), (4,5)]
    
    U = [[0] * 0 for _ in range(n*m)]  # Will resize
    V = [[0] * 0 for _ in range(m*p)]
    W = [[0] * 0 for _ in range(p*n)]
    
    # First, count total rank
    total_rank = 0
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += len(ig) * len(jg) * len(kg)  # Schoolbook
    
    # Initialize
    U = [[0] * total_rank for _ in range(n*m)]
    V = [[0] * total_rank for _ in range(m*p)]
    W = [[0] * total_rank for _ in range(p*n)]
    
    col = 0
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    # Use Strassen
                    for U_loc, V_loc, W_loc in strassen_terms:
                        for li, i in enumerate(ig):
                            for lj, j in enumerate(jg):
                                U[i*m+j][col] += U_loc[li*2+lj]
                        for lj, j in enumerate(jg):
                            for lk, k in enumerate(kg):
                                V[j*p+k][col] += V_loc[lj*2+lk]
                        for lk, k in enumerate(kg):
                            for li, i in enumerate(ig):
                                W[k*n+i][col] += W_loc[lk*2+li]
                        col += 1
                else:
                    # Schoolbook
                    for i in ig:
                        for j in jg:
                            for k in kg:
                                U[i*m+j][col] = 1
                                V[j*p+k][col] = 1
                                W[k*n+i][col] = 1
                                col += 1
    
    assert col == total_rank
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return schoolbook()
