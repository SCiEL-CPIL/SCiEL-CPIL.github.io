def construct():
    n, m, p = 2, 2, 14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    u_forms = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # Strassen linear forms for B (local indices 0: B00, 1: B01, 2: B10, 3: B11)
    # Global B indices for block k:
    # B00: 2k, B01: 2k+1, B10: 14+2k, B11: 15+2k
    v_forms = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # Strassen contributions to C (local indices 0: C00, 1: C01, 2: C10, 3: C11)
    # Global C indices for block k:
    # C00: 4k, C01: 4k+2, C10: 4k+1, C11: 4k+3
    # Note: W index is k*n + i. 
    # C[i, k] -> index k*n + i.
    # C00 (i=0, k=2k) -> 2k*2 + 0 = 4k
    # C01 (i=0, k=2k+1) -> (2k+1)*2 + 0 = 4k+2
    # C10 (i=1, k=2k) -> 2k*2 + 1 = 4k+1
    # C11 (i=1, k=2k+1) -> (2k+1)*2 + 1 = 4k+3
    
    w_forms = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    col = 0
    for k in range(7):
        for s in range(7):
            # U
            for i in range(4):
                U[i][col] = u_forms[s][i]
            
            # V
            # Local B indices: 0->2k, 1->2k+1, 2->14+2k, 3->15+2k
            b_indices = [2*k, 2*k+1, 14+2*k, 15+2*k]
            for l in range(4):
                V[b_indices[l]][col] = v_forms[s][l]
            
            # W
            # Local C indices: 0->4k, 1->4k+2, 2->4k+1, 3->4k+3
            c_indices = [4*k, 4*k+2, 4*k+1, 4*k+3]
            for l in range(4):
                W[c_indices[l]][col] = w_forms[s][l]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
