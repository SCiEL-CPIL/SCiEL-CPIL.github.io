"""Matrix multiplication 2,2,15 tensor decomposition."""
DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W with rank 53
    # 7 pairs of columns: (0,1), (2,3), ..., (12,13)
    # 1 single column: 14
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # (4, 53)
    V = [[0] * rank for _ in range(m * p)]  # (30, 53)
    W = [[0] * rank for _ in range(p * n)]  # (30, 53)
    
    col = 0
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices for a pair (k, k+1): 
    #   B0k, B1k, B0(k+1), B1(k+1)
    #   Global B indices: j*p + k
    #   j=0, k=k -> 0*p + k = k
    #   j=1, k=k -> 1*p + k = p + k
    #   j=0, k=k+1 -> k+1
    #   j=1, k=k+1 -> p + k + 1
    
    # C indices for a pair (k, k+1):
    #   C0k, C1k, C0(k+1), C1(k+1)
    #   Global W indices: k*n + i
    #   i=0, k=k -> k
    #   i=1, k=k -> n + k
    #   i=0, k=k+1 -> k+1
    #   i=1, k=k+1 -> n + k + 1
    
    strassen_A = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # For a pair of columns k, k+1
    # B00 = B0k, B01 = B0(k+1), B10 = B1k, B11 = B1(k+1)
    # M1: B0k + B1(k+1) -> V[0*p+k] += 1, V[1*p+k+1] += 1
    # M2: B0k -> V[0*p+k] += 1
    # M3: B0(k+1) - B1(k+1) -> V[0*p+k+1] += 1, V[1*p+k+1] += -1
    # M4: B1k - B0k -> V[1*p+k] += 1, V[0*p+k] += -1
    # M5: B1(k+1) -> V[1*p+k+1] += 1
    # M6: B0k + B0(k+1) -> V[0*p+k] += 1, V[0*p+k+1] += 1
    # M7: B1k + B1(k+1) -> V[1*p+k] += 1, V[1*p+k+1] += 1
    
    strassen_B = [
        {0: 1, p+1: 1}, # M1
        {0: 1}, # M2
        {1: 1, p+1: -1}, # M3
        {p: 1, 0: -1}, # M4
        {p+1: 1}, # M5
        {0: 1, 1: 1}, # M6
        {p: 1, p+1: 1} # M7
    ]
    
    # C reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C0k is index k in W (i=0, col=k)
    # C1k is index n+k in W (i=1, col=k)
    # C0(k+1) is index k+1 in W
    # C1(k+1) is index n+k+1 in W
    
    strassen_C = [
        {0: 1, p: 1, n+1: -1, n+p: 1}, # M1: C00, C10? No.
        # Let's map carefully.
        # C00 (i=0, k=k) -> W[k]
        # C10 (i=1, k=k) -> W[n+k]
        # C01 (i=0, k=k+1) -> W[k+1]
        # C11 (i=1, k=k+1) -> W[n+k+1]
        
        # M1: C00 + C11 -> W[k] += 1, W[n+k+1] += 1
        # M2: C10 + C11? No.
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes to C10 (+1) and C11 (-1).
        # M4 contributes to C00 (+1) and C10 (+1).
        # M5 contributes to C00 (-1) and C01 (+1).
        # M7 contributes to C00 (+1).
        # M3 contributes to C01 (+1) and C11 (+1).
        # M6 contributes to C11 (+1).
        
        # Let's re-verify Strassen reconstruction.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00 (+1), C11 (+1)
        # M2: C10 (+1), C11 (-1)
        # M3: C01 (+1), C11 (+1)
        # M4: C00 (+1), C10 (+1)
        # M5: C00 (-1), C01 (+1)
        # M6: C11 (+1)
        # M7: C00 (+1)
    ]
    
    # Correct W mappings for a pair starting at column k
    # W index for C[i, k'] is k'*n + i
    # k' = k or k+1
    # i = 0 or 1
    
    # M1: C00 (k, 0) -> k*n+0 = k; C11 (k+1, 1) -> (k+1)*n+1 = n+k+1
    # M2: C10 (k, 1) -> k*n+1 = n+k; C11 (k+1, 1) -> n+k+1
    # M3: C01 (k+1, 0) -> (k+1)*n+0 = k+1; C11 (k+1, 1) -> n+k+1
    # M4: C00 (k, 0) -> k; C10 (k, 1) -> n+k
    # M5: C00 (k, 0) -> k; C01 (k+1, 0) -> k+1
    # M6: C11 (k+1, 1) -> n+k+1
    # M7: C00 (k, 0) -> k
    
    strassen_W = [
        {k: 1, n+k+1: 1}, # M1
        {n+k: 1, n+k+1: -1}, # M2
        {k+1: 1, n+k+1: 1}, # M3
        {k: 1, n+k: 1}, # M4
        {k: -1, k+1: 1}, # M5
        {n+k+1: 1}, # M6
        {k: 1} # M7
    ]
    
    for pair_idx in range(7):
        k = pair_idx * 2
        for r in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    val = strassen_A[r][i * 2 + j]
                    if val != 0:
                        U[i * m + j][col] += val
            
            # V
            # B indices are relative to the pair.
            # 0 -> B0k -> global 0*p + k = k
            # 1 -> B0(k+1) -> global 0*p + k+1 = k+1
            # p -> B1k -> global 1*p + k = p + k
            # p+1 -> B1(k+1) -> global 1*p + k+1 = p + k + 1
            
            b_map = strassen_B[r]
            for b_rel, val in b_map.items():
                # b_rel is 0, 1, p, p+1
                # Map to global index
                if b_rel == 0:
                    idx = k
                elif b_rel == 1:
                    idx = k + 1
                elif b_rel == p:
                    idx = p + k
                elif b_rel == p + 1:
                    idx = p + k + 1
                else:
                    continue
                V[idx][col] += val
                
            # W
            w_map = strassen_W[r]
            # The keys in strassen_W were defined with k as a variable.
            # We need to substitute k with the current k.
            # My strassen_W definition above used 'k' as a symbol.
            # Let's re-evaluate strassen_W for current k.
            
            # M1: {k: 1, n+k+1: 1}
            # M2: {n+k: 1, n+k+1: -1}
            # M3: {k+1: 1, n+k+1: 1}
            # M4: {k: 1, n+k: 1}
            # M5: {k: -1, k+1: 1}
            # M6: {n+k+1: 1}
            # M7: {k: 1}
            
            if r == 0:
                W[k][col] += 1
                W[n+k+1][col] += 1
            elif r == 1:
                W[n+k][col] += 1
                W[n+k+1][col] -= 1
            elif r == 2:
                W[k+1][col] += 1
                W[n+k+1][col] += 1
            elif r == 3:
                W[k][col] += 1
                W[n+k][col] += 1
            elif r == 4:
                W[k][col] -= 1
                W[k+1][col] += 1
            elif r == 5:
                W[n+k+1][col] += 1
            elif r == 6:
                W[k][col] += 1
                
            col += 1
            
    # Last column k=14
    # Schoolbook: 4 terms
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # Term 1: A00 B0k -> C0k
    U[0][col] = 1 # A00
    V[k][col] = 1 # B0k (k=14)
    W[k][col] = 1 # C0k
    col += 1
    
    # Term 2: A01 B1k -> C0k
    U[1][col] = 1 # A01
    V[p+k][col] = 1 # B1k
    W[k][col] = 1 # C0k
    col += 1
    
    # Term 3: A10 B0k -> C1k
    U[2][col] = 1 # A10
    V[k][col] = 1 # B0k
    W[n+k][col] = 1 # C1k
    col += 1
    
    # Term 4: A11 B1k -> C1k
    U[3][col] = 1 # A11
    V[p+k][col] = 1 # B1k
    W[n+k][col] = 1 # C1k
    col += 1
    
    return {"U": U, "V": V, "W": W}
