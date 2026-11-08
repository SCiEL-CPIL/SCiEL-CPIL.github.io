import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen for 2x2x2 blocks and rank 1 for the remaining 2x1x10 part.
    # Total rank: 5 * 7 (for 2x2x10 part) + 1 (for 2x1x10 part) = 36.
    
    # Initialize matrices
    # U: (n*m, rank) = (6, 36)
    # V: (m*p, rank) = (30, 36)
    # W: (p*n, rank) = (20, 36)
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: (i, j) where i in {0,1}, j in {0,1} relative to block
    # B indices: (j, k) where j in {0,1}, k in {0,1} relative to block
    # C indices: (i, k)
    
    # U coefficients for A (a,b,c,d)
    # a: (0,0), b: (0,1), c: (1,0), d: (1,1)
    U_M = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [0, 0, 1, -1],  # M6: c-a
        [0, 1, 0, -1]   # M7: b-d
    ]
    
    # V coefficients for B (e,f,g,h)
    # e: (0,0), f: (0,1), g: (1,0), h: (1,1)
    V_M = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [0, -1, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1]    # M7: g+h
    ]
    
    # W coefficients for C (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_M = [
        [1, 0, 0, 1],   # M1
        [-1, 0, 1, 0],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # 5 blocks of 2x2x2 for A[0:2, 0:2] and B[0:2, 0:10]
    for block_idx in range(5):
        k_offset = block_idx * 2
        for s in range(7):
            # A indices: i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = U_M[s][i * 2 + j]
            
            # B indices: j in {0,1}, k in {k_offset, k_offset+1}
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_offset + k), col] = V_M[s][j * 2 + k]
            
            # C indices: i in {0,1}, k in {k_offset, k_offset+1}
            for i in range(2):
                for k in range(2):
                    W[(k_offset + k) * n + i, col] = W_M[s][i * 2 + k]
            
            col += 1
    
    # 1 block of 2x1x10 for A[0:2, 2:3] and B[2:3, 0:10]
    # Rank 1
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..9}
    # C indices: i in {0,1}, k in {0..9}
    
    for i in range(2):
        U[i * m + 2, col] = 1.0
        
    for k in range(10):
        V[2 * p + k, col] = 1.0
        
    for i in range(2):
        for k in range(10):
            W[k * n + i, col] = 1.0
            
    col += 1
    
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
