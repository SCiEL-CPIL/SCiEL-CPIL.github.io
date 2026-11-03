import time

def construct():
    deadline = time.time() + 140.0
    n, m, p = 3, 5, 6
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def add_schoolbook(i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        add_term(u, v, w)
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        """Strassen rank-7 for 2x2x2 sub-block.
        A[i,j] for i in {i0,i1}, j in {j0,j1}
        B[j,k] for j in {j0,j1}, k in {k0,k1}
        C[i,k] for i in {i0,i1}, k in {k0,k1}
        W convention: W[k*n+i, c]
        """
        # m0 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        u0 = {(i0, j0): 1, (i1, j1): 1}
        v0 = {(j0, k0): 1, (j1, k1): 1}
        w0 = {(k0, i0): 1, (k1, i1): 1}
        add_term(u0, v0, w0)
        
        # m1 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        u1 = {(i1, j0): 1, (i1, j1): 1}
        v1 = {(j0, k0): 1}
        w1 = {(k0, i1): 1, (k1, i1): -1}
        add_term(u1, v1, w1)
        
        # m2 = A[i0,j0](B[j1,k0]-B[j1,k1])
        u2 = {(i0, j0): 1}
        v2 = {(j1, k0): 1, (j1, k1): -1}
        w2 = {(k0, i0): 1}
        add_term(u2, v2, w2)
        
        # m3 = A[i1,j1](B[j0,k1]-B[j0,k0])
        u3 = {(i1, j1): 1}
        v3 = {(j0, k1): 1, (j0, k0): -1}
        w3 = {(k1, i1): 1}
        add_term(u3, v3, w3)
        
        # m4 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        u4 = {(i0, j0): 1, (i0, j1): 1}
        v4 = {(j1, k1): 1}
        w4 = {(k1, i1): 1, (k0, i0): -1}
        add_term(u4, v4, w4)
        
        # m5 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        u5 = {(i1, j0): 1, (i0, j0): -1}
        v5 = {(j0, k0): 1, (j0, k1): 1}
        w5 = {(k1, i1): 1}
        add_term(u5, v5, w5)
        
        # m6 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        u6 = {(i0, j1): 1, (i1, j1): -1}
        v6 = {(j1, k0): 1, (j1, k1): 1}
        w6 = {(k0, i0): 1, (k1, i1): -1}
        add_term(u6, v6, w6)
    
    def add_1x2x2(i, j0, j1, k0, k1):
        """Rank-2 decomposition for 1x2x2 sub-block.
        A[i,j] for j in {j0,j1}
        B[j,k] for j in {j0,j1}, k in {k0,k1}
        C[i,k] for k in {k0,k1}
        """
        # C[i,k0] += A[i,j0]*B[j0,k0]
        add_term({(i, j0): 1}, {(j0, k0): 1}, {(k0, i): 1})
        # C[i,k0] += A[i,j1]*B[j1,k0]; C[i,k1] += A[i,j1]*B[j1,k1]
        # Actually for 1x2x2, we need 2 terms:
        # term1: A[i,j0] * B[j0,k0] contributes to C[i,k0]
        # term2: A[i,j0] * B[j0,k1] contributes to C[i,k1]
        # Wait, that's 4 terms for schoolbook. Rank of 1x2x2 is 2.
        # 
        # C[i,k0] = A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
        # C[i,k1] = A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]
        #
        # Rank 2: 
        # m0 = A[i,j0]*(B[j0,k0] + B[j0,k1]) -> contributes to both C[i,k0] and C[i,k1]
        # m1 = A[i,j1]*(B[j1,k0] + B[j1,k1]) -> contributes to both
        # But then C[i,k0] = m0 + m1 - (something)? No...
        #
        # Actually: 
        # m0 = A[i,j0] * B[j0,k0]
        # m1 = A[i,j1] * B[j1,k0]  
        # m2 = A[i,j0] * B[j0,k1]
        # m3 = A[i,j1] * B[j1,k1]
        # That's rank 4 (schoolbook).
        #
        # But rank of 1x2x2 tensor: T[j,k] = A[j]*B[j,k] for j in {0,1}, k in {0,1}
        # This is a 2x2 matrix = diag(A) @ B. The tensor rank is at most 2.
        # 
        # m0 = A[0] * (B[0,0] + B[0,1])
        # m1 = A[1] * (B[1,0] + B[1,1])
        # C[0] = m0 + m1 - ??? 
        #
        # Hmm, let me think again. T[j,k] where j indexes A and B's middle, k indexes output.
        # T[j,k] = A[j] * B[j,k]
        # 
        # As a tensor in (A, B, C) space:
        # sum_c U[j,c] * V[j,k,c] * W[k,c]... no, the factorization is:
        # T[j,k] = sum_c U_A[j,c] * U_B[j,k,c] * U_C[k,c]
        #
        # Wait, I'm confusing myself. In the full problem, the tensor is T[i,j,k] where:
        # i = row of A / row of C (size n=3)
        # j = col of A / row of B (size m=5)  
        # k = col of B / col of C (size p=6)
        #
        # For a 1x2x2 sub-block with fixed i, j in {j0,j1}, k in {k0,k1}:
        # T[i,j,k] = delta (since we're decomposing the identity tensor)
        # 
        # The factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
        #
        # For the 1x2x2 case, we need:
        # For j=j0,k=k0: sum_c U[i*m+j0,c]*V[j0*p+k0,c]*W[k0*n+i,c] = 1
        # For j=j0,k=k1: sum_c U[i*m+j0,c]*V[j0*p+k1,c]*W[k1*n+i,c] = 1
        # For j=j1,k=k0: sum_c U[i*m+j1,c]*V[j1*p+k0,c]*W[k0*n+i,c] = 1
        # For j=j1,k=k1: sum_c U[i*m+j1,c]*V[j1*p+k1,c]*W[k1*n+i,c] = 1
        #
        # And it must be 0 for all other (i',j',k') combinations.
        #
        # Rank 2 construction:
        # c=0: U[i*m+j0]=1, V[j0*p+k0]=1, V[j0*p+k1]=1, W[k0*n+i]=1, W[k1*n+i]=1
        #       This gives: T[i,j0,k0] += 1, T[i,j0,k1] += 1
        # c=1: U[i*m+j1]=1, V[j1*p+k0]=1, V[j1*p+k1]=1, W[k0*n+i]=1, W[k1*n+i]=1
        #       This gives: T[i,j1,k0] += 1, T[i,j1,k1] += 1
        #
        # But we need T[i,j0,k0]=1, T[i,j0,k1]=1, T[i,j1,k0]=1, T[i,j1,k1]=1
        # With the above: T[i,j0,k0]=1, T[i,j0,k1]=1, T[i,j1,k0]=1, T[i,j1,k1]=1 ✓
        #
        # But wait, we also need it to be 0 for other (i',j',k'). Since U only has non-zero at (i,j0) and (i,j1),
        # and W only has non-zero at (k0,i) and (k1,i), this should be fine as long as no other terms
        # interfere. Since each term is independent, this works!
        
        # Term 1: covers j=j0, both k values
        add_term({(i, j0): 1}, {(j0, k0): 1, (j0, k1): 1}, {(k0, i): 1, (k1, i): 1})
        # Term 2: covers j=j1, both k values
        add_term({(i, j1): 1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i): 1, (k1, i): 1})
    
    def add_1x1x2(i, j, k0, k1):
        """Rank-1... wait, 1x1x2 needs rank 2? Let me check.
        T[i,j,k0] = 1, T[i,j,k1] = 1
        One term: U[i*m+j]=1, V[j*p+k0]=1, V[j*p+k1]=1, W[k0*n+i]=1, W[k1*n+i]=1
        This gives T[i,j,k0]=1, T[i,j,k1]=1. Rank 1!
        """
        add_term({(i, j): 1}, {(j, k0): 1, (j, k1): 1}, {(k0, i): 1, (k1, i): 1})
    
    def add_2x1x2(i0, i1, j, k0, k1):
        """2x1x2 sub-block: A[i,j] for i in {i0,i1}, B[j,k] for k in {k0,k1}, C[i,k]
        Rank 2:
        c=0: U[i0*m+j]=1, U[i1*m+j]=1, V[j*p+k0]=1, V[j*p+k1]=1, W[k0*n+i0]=1, W[k0*n+i1]=1, W[k1*n+i0]=1, W[k1*n+i1]=1
        This gives all 4 entries = 1. But we need each to be exactly 1, and this gives each as 1. ✓
        Actually wait: T[i0,j,k0] = U[i0*m+j]*V[j*p+k0]*W[k0*n+i0] = 1*1*1 = 1 ✓
        T[i0,j,k1] = U[i0*m+j]*V[j*p+k1]*W[k1*n+i0] = 1*1*1 = 1 ✓
        T[i1,j,k0] = U[i1*m+j]*V[j*p+k0]*W[k0*n+i1] = 1*1*1 = 1 ✓
        T[i1,j,k1] = U[i1*m+j]*V[j*p+k1]*W[k1*n+i1] = 1*1*1 = 1 ✓
        Rank 1!
        """
        add_term({(i0, j): 1, (i1, j): 1}, {(j, k0): 1, (j, k1): 1}, 
                 {(k0, i0): 1, (k0, i1): 1, (k1, i0): 1, (k1, i1): 1})
    
    # Main construction:
    # Split A (3x5) rows into [2, 1]: rows 0-1, row 2
    # Split B (5x6) cols into [2, 2, 2]: cols 0-1, 2-3, 4-5
    # Split B (5x6) rows into [2, 2, 1]: rows 0-1, 2-3, 4
    
    # For top 2 rows of A (i in {0,1}):
    #   For each column block of B (k in {0,1}, {2,3}, {4,5}):
    #     For each row block of B (j in {0,1}, {2,3}, {4}):
    #       If both i-block and j-block and k-block are size 2: Strassen
    #       Else: appropriate sub-decomposition
    
    # Let's enumerate all sub-problems:
    # i_blocks: [[0,1], [2]]
    # j_blocks: [[0,1], [2,3], [4]]  
    # k_blocks: [[0,1], [2,3], [4,5]]
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                if len(i_blk) == 2 and len(j_blk) == 2 and len(k_blk) == 2:
                    # 2x2x2: Strassen
                    i0, i1 = i_blk
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    add_strassen(i0, i1, j0, j1, k0, k1)
                elif len(i_blk) == 2 and len(j_blk) == 1 and len(k_blk) == 2:
                    # 2x1x2
                    i0, i1 = i_blk
                    j = j_blk[0]
                    k0, k1 = k_blk
                    add_2x1x2(i0, i1, j, k0, k1)
                elif len(i_blk) == 1 and len(j_blk) == 2 and len(k_blk) == 2:
                    # 1x2x2
                    i = i_blk[0]
                    j0, j1 = j_blk
                    k0, k1 = k_blk
                    add_1x2x2(i, j0, j1, k0, k1)
                elif len(i_blk) == 1 and len(j_blk) == 1 and len(k_blk) == 2:
                    # 1x1x2
                    i = i_blk[0]
                    j = j_blk[0]
                    k0, k1 = k_blk
                    add_1x1x2(i, j, k0, k1)
                else:
                    # Fallback to schoolbook for any other combination
                    for i in i_blk:
                        for j in j_blk:
                            for k in k_blk:
                                add_schoolbook(i, j, k)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return _schoolbook_fallback(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
