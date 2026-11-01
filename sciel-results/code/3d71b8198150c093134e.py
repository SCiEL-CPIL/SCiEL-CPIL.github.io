import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen forms for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # u vectors are for A (indices 0:a, 1:b, 2:c, 3:d)
    # v vectors are for B (local indices 0:e, 1:f, 2:g, 3:h)
    # w vectors are for C (local indices 0:C11, 1:C12, 2:C21, 3:C22)
    
    u_forms = [
        [1, 0, 0, 1], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [0, 0, 1, -1],# m6: c-a
        [0, 1, 0, -1] # m7: b-d
    ]
    
    v_forms = [
        [1, 1, 1, 1], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1],# m3: f-h
        [0, 0, 1, -1],# m4: g-e
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    
    w_forms = [
        [1, 0, 0, -1], # m1: C11+=1, C22-=1
        [0, 0, 1, -1], # m2: C21+=1, C22-=1
        [0, 1, 0, 1],  # m3: C12+=1, C22+=1
        [1, 0, 1, 0],  # m4: C11+=1, C21+=1
        [-1, 1, 0, 0], # m5: C11-=1, C12+=1
        [0, 0, 0, 1],  # m6: C22+=1
        [1, 0, 0, 0]   # m7: C11+=1
    ]
    
    num_blocks = p // 2 # 3 blocks
    
    for k in range(num_blocks):
        for s in range(7):
            r = k * 7 + s
            
            # U: same for all blocks
            for i in range(4):
                U[i][r] = u_forms[s][i]
                
            # V: place v_forms[s] into the k-th block of V
            # V has shape (12, 21).
            # Block k corresponds to rows 4k to 4k+3.
            # Local indices: 0:e, 1:f, 2:g, 3:h
            # Global indices: 4k, 4k+1, 4k+2, 4k+3
            for i in range(4):
                V[4*k + i][r] = v_forms[s][i]
                
            # W: place w_forms[s] into the k-th block of W
            # W has shape (12, 21).
            # Block k corresponds to rows 4k to 4k+3.
            # Local indices: 0:C11, 1:C12, 2:C21, 3:C22
            # Global indices: 4k, 4k+1, 4k+2, 4k+3
            for i in range(4):
                W[4*k + i][r] = w_forms[s][i]
                
    return {"U": U, "V": V, "W": W}
