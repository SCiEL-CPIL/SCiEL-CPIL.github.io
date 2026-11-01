DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2 block
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11]
    # w: [C00, C10, C01, C11] -> Note: W index is k*n+i.
    # C00: k=0, i=0 -> idx 0
    # C10: k=0, i=1 -> idx 1
    # C01: k=1, i=0 -> idx 2
    # C11: k=1, i=1 -> idx 3
    # So w vector order: [C00, C10, C01, C11]
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 1, 0], # M5
        [0, 1, 0, 1], # M6
        [1, 0, 1, 0]  # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 1, 0, -1],# M2 -> C10, -C11
        [0, 0, 1, 1], # M3 -> C01, C11
        [1, 1, 0, 0], # M4 -> C00, C10
        [-1, 0, 1, 0],# M5 -> -C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Map block indices to global V and W indices
    # B is 2x6. V index = j*6 + k.
    # Block b (0, 1, 2) corresponds to k offsets 2*b, 2*b+1.
    # B00 -> j=0, k=2*b -> V[0*6 + 2*b]
    # B01 -> j=0, k=2*b+1 -> V[0*6 + 2*b+1]
    # B10 -> j=1, k=2*b -> V[1*6 + 2*b]
    # B11 -> j=1, k=2*b+1 -> V[1*6 + 2*b+1]
    
    # C is 2x6. W index = k*2 + i.
    # Block b corresponds to k offsets 2*b, 2*b+1.
    # C00 -> k=2*b, i=0 -> W[2*b*2 + 0]
    # C10 -> k=2*b, i=1 -> W[2*b*2 + 1]
    # C01 -> k=2*b+1, i=0 -> W[(2*b+1)*2 + 0]
    # C11 -> k=2*b+1, i=1 -> W[(2*b+1)*2 + 1]
    
    col = 0
    for b in range(3):
        for s in range(7):
            u = strassen_u[s]
            v = strassen_v[s]
            w = strassen_w[s]
            
            # U indices: i*m + j
            # u[0] is A00 (i=0, j=0) -> idx 0
            # u[1] is A01 (i=0, j=1) -> idx 1
            # u[2] is A10 (i=1, j=0) -> idx 2
            # u[3] is A11 (i=1, j=1) -> idx 3
            U[0][col] = u[0]
            U[1][col] = u[1]
            U[2][col] = u[2]
            U[3][col] = u[3]
            
            # V indices
            # v[0] is B00 (j=0, k=2*b) -> idx 0*6 + 2*b
            # v[1] is B01 (j=0, k=2*b+1) -> idx 0*6 + 2*b+1
            # v[2] is B10 (j=1, k=2*b) -> idx 1*6 + 2*b
            # v[3] is B11 (j=1, k=2*b+1) -> idx 1*6 + 2*b+1
            V[0*6 + 2*b][col] = v[0]
            V[0*6 + 2*b+1][col] = v[1]
            V[1*6 + 2*b][col] = v[2]
            V[1*6 + 2*b+1][col] = v[3]
            
            # W indices
            # w[0] is C00 (k=2*b, i=0) -> idx 2*b*2 + 0
            # w[1] is C10 (k=2*b, i=1) -> idx 2*b*2 + 1
            # w[2] is C01 (k=2*b+1, i=0) -> idx (2*b+1)*2 + 0
            # w[3] is C11 (k=2*b+1, i=1) -> idx (2*b+1)*2 + 1
            W[2*b*2 + 0][col] = w[0]
            W[2*b*2 + 1][col] = w[1]
            W[(2*b+1)*2 + 0][col] = w[2]
            W[(2*b+1)*2 + 1][col] = w[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
