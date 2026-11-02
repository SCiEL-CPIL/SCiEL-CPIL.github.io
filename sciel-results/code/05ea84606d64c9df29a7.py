import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (22, rank)
    # W: (p*n, rank) = (22, rank)
    
    rank = 39
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # 5 blocks of 2x2 for B
    for s in range(5):
        k0 = 2 * s
        
        # Term 1: (a+d)(e+h) -> M1
        # c11 += M1, c22 += M1
        U[:, col] = [1, 0, 0, 1]
        V[0*11 + k0 + 0, col] = 1
        V[1*11 + k0 + 1, col] = 1
        W[(k0 + 0)*2 + 0, col] = 1  # c11
        W[(k0 + 1)*2 + 1, col] = 1  # c22
        col += 1
        
        # Term 2: (c+d)e -> M2
        # c21 += M2, c22 -= M2
        U[:, col] = [0, 0, 1, 1]
        V[0*11 + k0 + 0, col] = 1
        W[(k0 + 0)*2 + 1, col] = 1  # c21
        W[(k0 + 1)*2 + 1, col] = -1 # c22
        col += 1
        
        # Term 3: a(f-h) -> M3
        # c12 += M3, c22 += M3
        U[:, col] = [1, 0, 0, 0]
        V[0*11 + k0 + 1, col] = 1
        V[1*11 + k0 + 0, col] = -1
        W[(k0 + 1)*2 + 0, col] = 1  # c12
        W[(k0 + 1)*2 + 1, col] = 1  # c22
        col += 1
        
        # Term 4: d(g-e) -> M4
        # c11 += M4, c22 += M4
        U[:, col] = [0, 0, 0, 1]
        V[0*11 + k0 + 0, col] = -1
        V[1*11 + k0 + 0, col] = 1
        W[(k0 + 0)*2 + 0, col] = 1  # c11
        W[(k0 + 1)*2 + 1, col] = 1  # c22 (FIXED: was (k0+0)*2+1)
        col += 1
        
        # Term 5: (a+b)h -> M5
        # c11 -= M5, c12 += M5
        U[:, col] = [1, 1, 0, 0]
        V[1*11 + k0 + 1, col] = 1
        W[(k0 + 0)*2 + 0, col] = -1 # c11
        W[(k0 + 1)*2 + 0, col] = 1  # c12
        col += 1
        
        # Term 6: (c-a)(e+f) -> M6
        # c21 += M6
        U[:, col] = [0, 0, 1, -1]
        V[0*11 + k0 + 0, col] = 1
        V[0*11 + k0 + 1, col] = 1
        W[(k0 + 0)*2 + 1, col] = 1  # c21 (FIXED: was (k0+1)*2+1)
        col += 1
        
        # Term 7: (b-d)(g+h) -> M7
        # c11 += M7
        U[:, col] = [0, 1, 0, -1]
        V[1*11 + k0 + 0, col] = 1
        V[1*11 + k0 + 1, col] = 1
        W[(k0 + 0)*2 + 0, col] = 1  # c11
        col += 1

    # Last block 2x1 for B (column 10)
    k0 = 10
    # Term 8: a*e
    U[:, col] = [1, 0, 0, 0]
    V[0*11 + k0, col] = 1
    W[k0*2 + 0, col] = 1
    col += 1
    
    # Term 9: b*g
    U[:, col] = [0, 1, 0, 0]
    V[1*11 + k0, col] = 1
    W[k0*2 + 0, col] = 1
    col += 1
    
    # Term 10: c*e
    U[:, col] = [0, 0, 1, 0]
    V[0*11 + k0, col] = 1
    W[k0*2 + 1, col] = 1
    col += 1
    
    # Term 11: d*g
    U[:, col] = [0, 0, 0, 1]
    V[1*11 + k0, col] = 1
    W[k0*2 + 1, col] = 1
    col += 1
    
    # Verify shapes
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        raise ValueError("Shape mismatch")
        
    return {"U": U, "V": V, "W": W}
