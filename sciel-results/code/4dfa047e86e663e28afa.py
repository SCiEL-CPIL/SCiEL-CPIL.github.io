import time

def construct():
    """
    Rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm independently on each of the
    four 2x2 block products:
      A1*B11, A2*B21, A1*B12, A2*B22
    
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing (trace convention W[k*n+i]):
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i
    """
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin under 150s kill
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen's 7-multiplication patterns for 2x2 matrix multiplication
    # A entries indexed as: a11=0, a12=1, a21=2, a22=3
    # B entries indexed as: b11=0, b12=1, b21=2, b22=3
    # C entries indexed as: c11=0, c12=1, c21=2, c22=3
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)
    # M3 = (a11)(b12-b22)
    # M4 = (a22)(b21-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    strassen_patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    # Four block products:
    # Block 1: A1 * B11 -> C[:, 0:2] (C_left)
    #   A1 = A[:, 0:2]: rows i=0,1; cols j=0,1
    #   B11 = B[0:2, 0:2]: rows j=0,1; cols k=0,1
    #
    # Block 2: A2 * B21 -> C[:, 0:2] (C_left)
    #   A2 = A[:, 2:4]: rows i=0,1; cols j=2,3
    #   B21 = B[2:4, 0:2]: rows j=2,3; cols k=0,1
    #
    # Block 3: A1 * B12 -> C[:, 2:4] (C_right)
    #   A1 = A[:, 0:2]: rows i=0,1; cols j=0,1
    #   B12 = B[0:2, 2:4]: rows j=0,1; cols k=2,3
    #
    # Block 4: A2 * B22 -> C[:, 2:4] (C_right)
    #   A2 = A[:, 2:4]: rows i=0,1; cols j=2,3
    #   B22 = B[2:4, 2:4]: rows j=2,3; cols k=2,3
    
    # Block definitions: (a_base_j, b_base_j, b_base_k, c_base_k)
    blocks = [
        (0, 0, 0, 0),  # A1*B11 -> C[:, 0:2]
        (2, 2, 0, 0),  # A2*B21 -> C[:, 0:2]
        (0, 0, 2, 2),  # A1*B12 -> C[:, 2:4]
        (2, 2, 2, 2),  # A2*B22 -> C[:, 2:4]
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            # Fill U: A entries
            # local (r,c) where local_idx = r*2+c -> global (i=r, j=a_bj+c)
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j][col] = coeff
            
            # Fill V: B entries
            # local (r,c) where local_idx = r*2+c -> global (j=b_bj+r, k=b_bk+c)
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k][col] = coeff
            
            # Fill W: C entries (trace convention W[k*n+i])
            # local (r,c) where local_idx = r*2+c -> global (i=r, k=c_bk+c)
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    # Deadline guard (construction is O(1); this is a safety net)
    if time.time() > deadline:
        pass  # still return the completed, valid rank-28 result
    
    return {"U": U, "V": V, "W": W}
