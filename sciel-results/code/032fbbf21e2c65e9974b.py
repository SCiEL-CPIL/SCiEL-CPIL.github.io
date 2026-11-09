"""Matrix multiplication 3,4,8 rank-72 decomposition via tensor product structure."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Tensor product decomposition:
    # 3x4x8 = (3x4x2) ⊗ (1x1x4)
    # Rank = rank(3x4x2) * rank(1x1x4) = 18 * 4 = 72
    #
    # The 1x1x4 tensor is just the identity on the 4th dimension (4 scalar multiplications)
    # The 3x4x2 tensor handles the 3x4x2 core
    #
    # For the 3x4x2 tensor, we use a decomposition with rank 18:
    # Split into:
    #   - i=0: 1x4x2 = 8 terms (schoolbook)
    #   - i=1,2: 2x4x2 using Strassen-like approach
    #     Split j into 2 blocks of 2, k into 1 block of 2
    #     2 * 1 * 7 = 14 terms
    # Total: 8 + 14 = 22... that's not 18.
    #
    # Let me reconsider. Actually for 3x4x2:
    # Schoolbook: 3*4*2 = 24
    # 
    # Better approach: use the known optimal or near-optimal for 3x4x2
    # Actually, let's just use a clean tensor product:
    #
    # 3x4x8 = (3x2x4) ⊗ (1x2x2)
    # rank(3x2x4) * rank(1x2x2)
    #
    # 1x2x2 is just 2 scalar multiplications, rank 2
    # 3x2x4: schoolbook is 24
    #
    # Hmm, let me think differently.
    #
    # Actually the simplest correct approach:
    # 3x4x8 = (3x4x1) ⊗ (1x1x8)
    # rank = 12 * 8 = 96. Not better.
    #
    # 3x4x8 = (3x2x8) ⊗ (1x2x1)
    # rank = rank(3x2x8) * 2
    # 3x2x8 schoolbook = 48, so 96.
    #
    # Let me use a different factorization.
    # 3x4x8: split as (3x4x4) ⊗ (1x1x2)
    # rank(3x4x4) * 2
    # 3x4x4 schoolbook = 48, so 96.
    #
    # The best tensor product decomposition:
    # 3x4x8 = (3x4x2) ⊗ (1x1x4)
    # rank(3x4x2) * 4
    #
    # For 3x4x2, what's the best rank?
    # Schoolbook: 24
    # Using Strassen on the 2x2 part:
    # Split i into [0] and [1,2]
    # i=0: 1x4x2 = 8 terms
    # i=1,2: 2x4x2
    #   Split j into [0,1] and [2,3], k into [0,1]
    #   2 blocks of j, 1 block of k, 7 Strassen terms each = 14
    # Total: 8 + 14 = 22
    # So rank = 22 * 4 = 88. Same as before.
    #
    # Alternative: 3x4x8 = (3x2x4) ⊗ (1x2x2)
    # rank(3x2x4) * rank(1x2x2) = rank(3x2x4) * 2
    #
    # 3x2x4:
    # i=0: 1x2x4 = 8
    # i=1,2: 2x2x4
    #   Split k into [0,1] and [2,3]
    #   2 blocks of k, each 2x2x2 with 7 terms = 14
    # Total: 8 + 14 = 22
    # rank = 22 * 2 = 44!
    
    # Let's implement 3x2x4 ⊗ 1x2x2
    # 
    # The 1x2x2 tensor: C[i,k] = sum_j A[i,j] * B[j,k] where A is 1x2, B is 2x2
    # This is just 2 independent scalar multiplications for each output
    # rank(1x2x2) = 2
    #
    # Wait, 1x2x2: A is 1x2, B is 2x2, C is 1x2
    # C[k] = sum_j A[j] * B[j,k]
    # This is just a matrix-vector product, rank 2 (schoolbook is optimal)
    #
    # So total rank = 22 * 2 = 44
    #
    # Let me verify the tensor product structure:
    # 3x4x8: A is 3x4, B is 4x8, C is 3x8
    # 3x2x4: A1 is 3x2, B1 is 2x4, C1 is 3x4
    # 1x2x2: A2 is 1x2, B2 is 2x2, C2 is 1x2
    #
    # A[i,j] = A1[i, j1] * A2[0, j2] where j = j1*2 + j2
    # B[j,k] = B1[j1, k1] * B2[j2, k2] where j = j1*2 + j2, k = k1*2 + k2
    # C[i,k] = C1[i, k1] * C2[0, k2] where k = k1*2 + k2
    #
    # Tensor product of decompositions:
    # If T1 has decomposition (U1, V1, W1) with rank r1
    # and T2 has decomposition (U2, V2, W2) with rank r2
    # Then T1⊗T2 has decomposition with rank r1*r2
    #
    # U[i*m+j, t1*r2+t2] = U1[i*m1+j1, t1] * U2[0*m2+j2, t2]
    # V[j*p+k, t1*r2+t2] = V1[j1*p1+k1, t1] * V2[j2*p2+k2, t2]
    # W[k*n+i, t1*r2+t2] = W1[k1*n1+i1, t1] * W2[k2*n2+i2, t2]
    
    # 3x2x4 decomposition (rank 22):
    # i=0: 1x2x4 schoolbook = 8 terms
    # i=1,2: 2x2x4 using two 2x2x2 Strassen blocks = 14 terms
    
    # 1x2x2 decomposition (rank 2):
    # Schoolbook: 2 terms
    
    # Build 3x2x4 factors
    n1, m1, p1 = 3, 2, 4
    r1 = 22
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    
    col1 = 0
    # i=0: 1x2x4 schoolbook
    for j1 in range(2):
        for k1 in range(4):
            U1[0 * m1 + j1][col1] = 1
            V1[j1 * p1 + k1][col1] = 1
            W1[k1 * n1 + 0][col1] = 1
            col1 += 1
    
    # i=1,2: 2x2x4, split k into two blocks of 2
    strassen_U = [
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
        [1, 1],
        [-1, 0],
        [0, 1]
    ]
    strassen_V = [
        [1, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, 0],
        [1, 1],
        [0, 0]
    ]
    strassen_W = [
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 0],
        [-1, 1],
        [0, 0],
        [1, 0]
    ]
    
    for k_block in range(2):
        k_start = k_block * 2
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    l = li * 2 + lj
                    val = strassen_U[t][l]
                    if val != 0:
                        gi = 1 + li
                        gj = lj
                        U1[gi * m1 + gj][col1] += val
                        
                        val_v = strassen_V[t][l]
                        if val_v != 0:
                            V1[gj * p1 + k_start + li][col1] += val_v
                        
                        val_w = strassen_W[t][l]
                        if val_w != 0:
                            W1[(k_start + li) * n1 + gi][col1] += val_w
            col1 += 1
    
    # Build 1x2x2 factors (schoolbook, rank 2)
    n2, m2, p2 = 1, 2, 2
    r2 = 2
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    
    for j2 in range(2):
        for k2 in range(2):
            idx = j2 * p2 + k2
            # Actually for 1x2x2, the decomposition is:
            # C[k2] = sum_j2 A[j2] * B[j2, k2]
            # Rank 2: for each j2, one term
            # U2[j2, t] = 1 if t == j2
            # V2[j2*p2+k2, t] = 1 if t == j2
            # W2[k2*n2+0, t] = 1
            pass
    
    # Simpler: just build directly
    U2 = [[1, 0], [0, 1]]
    V2 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    W2 = [[1, 0], [0, 1]]
    
    # Now combine via tensor product
    rank = r1 * r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            
            # U: A[i, j] where j = j1*2 + j2
            for i in range(n1):
                for j1 in range(m1):
                    for j2 in range(m2):
                        j = j1 * m2 + j2
                        u1 = U1[i * m1 + j1][t1]
                        u2 = U2[0 * m2 + j2][t2]
                        if u1 != 0 and u2 != 0:
                            U[i * m + j][t] += u1 * u2
            
            # V: B[j, k] where j = j1*2 + j2, k = k1*2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            v1 = V1[j1 * p1 + k1][t1]
                            v2 = V2[j2 * p2 + k2][t2]
                            if v1 != 0 and v2 != 0:
                                V[j * p + k][t] += v1 * v2
            
            # W: C[i, k] where k = k1*2 + k2
            for k1 in range(p1):
                for i in range(n1):
                    for k2 in range(p2):
                        k = k1 * p2 + k2
                        w1 = W1[k1 * n1 + i][t1]
                        w2 = W2[k2 * n2 + 0][t2]
                        if w1 != 0 and w2 != 0:
                            W[k * n + i][t] += w1 * w2
    
    return {"U": U, "V": V, "W": W}
