# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    # Rank 25: 3 blocks of Strassen (rank 7) + 1 block of Schoolbook (rank 4)
    rank = 25
    
    # Initialize matrices with zeros
    # U: (n*m) x rank = 4 x 25
    # V: (m*p) x rank = 14 x 25
    # W: (p*n) x rank = 14 x 25
    # Note: W index is k*n + i. k is 0..6, i is 0..1.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Coefficients for 2x2 by 2x2
    # U (A) coeffs: [A00, A01, A10, A11]
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: -A00 + A10
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # V (B) coeffs: [B00, B01, B10, B11]
    # Mapping to global V index: V[j*p + k]
    # For block b, columns are 2b, 2b+1.
    # B00 -> j=0, k=2b -> V[0*p + 2b]
    # B01 -> j=0, k=2b+1 -> V[0*p + 2b+1]
    # B10 -> j=1, k=2b -> V[1*p + 2b]
    # B11 -> j=1, k=2b+1 -> V[1*p + 2b+1]
    strassen_V_local = [
        {0: 1, 3: 1},   # M1: B00 + B11
        {0: 1},         # M2: B00
        {1: 1, 3: -1},  # M3: B01 - B11
        {2: 1, 0: -1},  # M4: B10 - B00
        {3: 1},         # M5: B11
        {0: 1, 1: 1},   # M6: B00 + B01
        {2: 1, 3: 1}    # M7: B10 + B11
    ]
    
    # W (C) coeffs: [C00, C01, C10, C11]
    # Mapping to global W index: W[k*n + i]
    # For block b, columns k are 2b, 2b+1.
    # C00 -> k=2b, i=0 -> W[2b*n + 0]
    # C01 -> k=2b+1, i=0 -> W[(2b+1)*n + 0]
    # C10 -> k=2b, i=1 -> W[2b*n + 1]
    # C11 -> k=2b+1, i=1 -> W[(2b+1)*n + 1]
    strassen_W_local = [
        {0: 1, 3: 1},   # M1: C00 + C11
        {2: 1, 3: -1},  # M2: C10 - C11
        {1: 1, 3: 1},   # M3: C01 + C11
        {0: 1, 2: 1},   # M4: C00 + C10
        {0: -1, 1: 1},  # M5: -C00 + C01
        {3: 1},         # M6: C11
        {0: 1}          # M7: C00
    ]
    
    # Process 3 Strassen blocks (cols 0-1, 2-3, 4-5)
    for b in range(3):
        for r in range(7):
            # Fill U
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            
            # Fill V
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            
            # Fill W
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
            
    # Schoolbook for last block (col 6)
    # A is 2x2, B is 2x1 (only col 6 of B is used)
    # C is 2x1 (only col 6 of C is produced)
    # Terms: A00*B60, A01*B61, A10*B60, A11*B61
    # B indices: B[0, 6] and B[1, 6]
    # V indices: V[0*p + 6] and V[1*p + 6]
    # W indices: W[6*n + 0] and W[6*n + 1]
    
    schoolbook_U = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    schoolbook_V_local = [
        {0: 1}, # B[0,6]
        {1: 1}, # B[1,6]
        {0: 1}, # B[0,6]
        {1: 1}  # B[1,6]
    ]
    
    schoolbook_W_local = [
        {0: 1}, # C[0,6]
        {0: 1}, # C[0,6]
        {1: 1}, # C[1,6]
        {1: 1}  # C[1,6]
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
            
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx][col] += coeff
            
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
