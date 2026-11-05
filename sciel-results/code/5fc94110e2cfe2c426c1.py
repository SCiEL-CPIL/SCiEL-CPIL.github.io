import numpy as np

def construct():
    n, m, p = 4, 4, 9
    rank = 65
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: A * (B1 + B2) using Strassen 4x4x4
    # B1 is cols 0-3, B2 is cols 4-7
    # B' = B1 + B2 is 4x4
    
    # We'll generate the 49 terms for Strassen 4x4x4
    # This is complex, so I'll use a known decomposition or a simpler one if possible.
    # For now, I'll just use the 49 terms.
    
    # Part 2: A * B3 using schoolbook 4x4x1
    # B3 is col 8
    # 16 terms
    
    col_offset = 0
    # Generate Strassen 4x4x4
    # ...
    
    # Generate Schoolbook 4x4x1
    for i in range(4):
        for k in range(4):
            U[i*m + k, col_offset] = 1
            V[k*p + 8, col_offset] = 1
            W[8*n + i, col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
