import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses 6 blocks of 2 columns. Each block uses Strassen (7 terms).
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # M1 = (A00 + A11) * (B0,c0 + B1,c1)
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[0*p + c0][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = 1
        W[c1*n + 1][c] = 1
        
        # M2 = (A10 + A11) * B0,c0
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[0*p + c0][c] = 1
        W[c0*n + 1][c] = 1
        W[c1*n + 1][c] = -1
        
        # M3 = A00 * (B0,c1 - B1,c1)
        c = base + 2
        U[0][c] = 1
        V[0*p + c1][c] = 1
        V[1*p + c1][c] = -1
        W[c1*n + 0][c] = 1
        W[c1*n + 1][c] = 1
        
        # M4 = A11 * (B1,c0 - B0,c0)
        c = base + 3
        U[3][c] = 1
        V[1*p + c0][c] = 1
        V[0*p + c0][c] = -1
        W[c0*n + 0][c] = 1
        W[c0*n + 1][c] = 1
        
        # M5 = (A00 + A01) * B1,c1
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = -1
        W[c1*n + 0][c] = 1
        
        # M6 = (A10 - A00) * (B0,c0 + B0,c1)
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[0*p + c0][c] = 1
        V[0*p + c1][c] = 1
        W[c1*n + 1][c] = 1
        
        # M7 = (A01 - A11) * (B1,c0 + B1,c1)
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[1*p + c0][c] = 1
        V[1*p + c1][c] = 1
        W[c0*n + 0][c] = 1
    
    return {"U": U, "V": V, "W": W}
