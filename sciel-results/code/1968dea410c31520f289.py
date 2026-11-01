import numpy as np

def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The rank is minimal (35) because the tensor is the direct sum of 5 copies of the 
    2x2x2 matrix multiplication tensor, each of which has rank 7.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = 2, 2, 10
    rank = 35
    
    # U shape: (n*m, rank) = (4, 35)
    # V shape: (m*p, rank) = (20, 35)
    # W shape: (p*n, rank) = (20, 35)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Products P1..P7
    # U coefficients for A (indices 0:a, 1:b, 2:c, 3:d)
    strassen_U = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    # V coefficients for B (local indices 0:e, 1:f, 2:g, 3:h)
    strassen_V = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W coefficients for C (local indices 0:C00, 1:C01, 2:C10, 3:C11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1],  # P3
        [1, 0, 1, 0],  # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0]   # P7
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            # Fill U
            for i in range(4):
                U[i, col] = strassen_U[r][i]
            
            # Fill V
            # e -> V[2*b], f -> V[2*b+1], g -> V[10+2*b], h -> V[10+2*b+1]
            V[2*b, col] = strassen_V[r][0]
            V[2*b+1, col] = strassen_V[r][1]
            V[10+2*b, col] = strassen_V[r][2]
            V[10+2*b+1, col] = strassen_V[r][3]
            
            # Fill W
            # C00 -> W[2*b] (i=0, k=2b)
            # C01 -> W[2*b+1] (i=0, k=2b+1)
            # C10 -> W[10+2*b] (i=1, k=2b)
            # C11 -> W[10+2*b+1] (i=1, k=2b+1)
            W[2*b, col] = strassen_W[r][0]
            W[2*b+1, col] = strassen_W[r][1]
            W[10+2*b, col] = strassen_W[r][2]
            W[10+2*b+1, col] = strassen_W[r][3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
