import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Terms 0-6 correspond to M1-M7
    # U coefficients for A (indices 0,1,2,3 for a,b,c,d)
    # V coefficients for B (indices 0,1,2,3 for e,f,g,h)
    # W coefficients for C (indices 0,1,2,3 for c11, c21, c12, c22? No, index is k*n+i)
    
    # Let's define the patterns for one block
    # U: 4x7
    # V: 4x7 (local)
    # W: 4x7 (local)
    
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-f)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B local indices: 0:e, 1:f, 2:g, 3:h
    # C local indices: 
    # Global C index k*n+i.
    # Local k=0, i=0 -> 0
    # Local k=0, i=1 -> 1
    # Local k=1, i=0 -> 2
    # Local k=1, i=1 -> 3
    
    # W coefficients for C (local indices 0,1,2,3)
    # C00 (idx 0) = M1 + M4 - M5 + M7
    # C10 (idx 1) = M2 + M4
    # C01 (idx 2) = M3 + M5
    # C11 (idx 3) = M1 - M2 + M3 + M6
    
    U_pattern = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    V_pattern = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    W_pattern = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (idx 0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (idx 1)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (idx 2)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (idx 3)
    ]
    
    # Fill the global matrices
    # There are 5 blocks (p=10, m=2, so 5 blocks of width 2)
    for l in range(5):
        for t in range(7):
            r = 7 * l + t
            
            # U
            for i in range(4):
                U[i][r] = U_pattern[i][t]
                
            # V
            # Global B index: j * p + k
            # Local B index: j * 2 + k_local
            # k = 2*l + k_local
            # So global index = j * 10 + 2*l + k_local
            for j in range(2):
                for k_local in range(2):
                    local_idx = j * 2 + k_local
                    global_idx = j * p + (2 * l + k_local)
                    V[global_idx][r] = V_pattern[local_idx][t]
                    
            # W
            # Global C index: k * n + i
            # k = 2*l + k_local
            # i is local i
            for k_local in range(2):
                for i in range(2):
                    local_idx = k_local * 2 + i
                    global_idx = (2 * l + k_local) * n + i
                    W[global_idx][r] = W_pattern[local_idx][t]
                    
    return {"U": U, "V": V, "W": W}
