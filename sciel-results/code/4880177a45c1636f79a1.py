def construct():
    n, m, p = DIMENSIONS
    # Split p=6 into two blocks of size 3
    # Each block is a 2x3x3 multiplication
    # Rank of 2x3x3 = 13 (7 Strassen + 6 scalar)
    # Total rank = 2 * 13 = 26
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 6 x 26
    V = [[0] * rank for _ in range(m * p)]  # 18 x 26
    W = [[0] * rank for _ in range(p * n)]  # 12 x 26
    
    # Terms for a 2x3x3 block
    # A (2x3): i in 0..1, j in 0..2. flat = i*3+j
    # B (3x3): j in 0..2, k in 0..2. flat = j*3+k
    # C (2x3): i in 0..1, k in 0..2. flat = i*3+k
    # W convention: W[k*n+i, r]
    
    terms = []
    
    # Strassen terms for A_L (2x2, j=0,1) * B_T (2x2, j=0,1, k=0,1)
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 4: 1},
        'w': {0: 1, 3: 1}  # C00, C11
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {1: 1, 3: -1}  # C10, C11
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 4: -1},
        'w': {2: 1, 3: 1}  # C01, C11
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {3: 1, 0: -1},
        'w': {0: 1, 1: 1}  # C00, C10
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {4: 1},
        'w': {0: -1, 2: 1}  # C00, C01
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}  # C11
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {3: 1, 4: 1},
        'w': {0: 1}  # C00
    })
    
    # Scalar terms for A[:,2] * B[2,:]
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=6, B[2,1]=7, B[2,2]=8
    
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {6: 1},
        'w': {0: 1}  # C00
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {7: 1},
        'w': {2: 1}  # C01
    })
    # K3 = A02 * B22
    terms.append({
        'u': {2: 1},
        'v': {8: 1},
        'w': {4: 1}  # C02
    })
    # K4 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {6: 1},
        'w': {1: 1}  # C10
    })
    # K5 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {7: 1},
        'w': {3: 1}  # C11
    })
    # K6 = A12 * B22
    terms.append({
        'u': {5: 1},
        'v': {8: 1},
        'w': {5: 1}  # C12
    })
    
    # Fill the matrices for 2 blocks
    for block in range(2):
        for t_idx, term in enumerate(terms):
            col = block * 13 + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V
            # B indices in block: j in 0..2, k in 0..2
            # Global B index: j*p + (block*3 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            
            # W
            # C indices in block: i in 0..1, k in 0..2
            # Global C index: k*n + i
            # term['w'] keys are local C indices (i*3+k)
            for l_idx, val in term['w'].items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
