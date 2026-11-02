import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 Strassen decomposition
    # 3 blocks of 2x2 multiplication
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # u: A factors (4)
    # v: B factors (4)
    # w: C factors (4)
    # Order of A: 00, 01, 10, 11
    # Order of B: 00, 01, 10, 11
    # Order of C: 00, 01, 10, 11
    
    strassen = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 0, 0]),
        # M5
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 1, 0], [1, 0, 0, 0])
    ]
    
    col = 0
    for t in range(3):
        for u, v, w in strassen:
            # Map A indices
            for i in range(4):
                U[i][col] = u[i]
            
            # Map B indices
            # Block t uses columns 2t, 2t+1 of B
            # B index in full matrix: j*6 + k
            # B^{(t)}_{0,0} -> j=0, k=2t -> idx 2t
            # B^{(t)}_{0,1} -> j=0, k=2t+1 -> idx 2t+1
            # B^{(t)}_{1,0} -> j=1, k=2t -> idx 6+2t
            # B^{(t)}_{1,1} -> j=1, k=2t+1 -> idx 6+2t+1
            
            v_indices = [2*t, 2*t+1, 6+2*t, 6+2*t+1]
            for i in range(4):
                V[v_indices[i]][col] = v[i]
                
            # Map C indices
            # W index: k*n + i
            # C^{(t)}_{0,0} -> k=2t, i=0 -> idx 2t*2 + 0 = 4t
            # C^{(t)}_{0,1} -> k=2t+1, i=0 -> idx (2t+1)*2 + 0 = 4t+2
            # C^{(t)}_{1,0} -> k=2t, i=1 -> idx 2t*2 + 1 = 4t+1
            # C^{(t)}_{1,1} -> k=2t+1, i=1 -> idx (2t+1)*2 + 1 = 4t+3
            
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for i in range(4):
                W[w_indices[i]][col] = w[i]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
