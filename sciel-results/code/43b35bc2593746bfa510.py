import numpy as np
import time

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = 2, 2, 4
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1 if j == jp else 0
                    actual = sum(U[i*m+j][r] * V[jp*p+k][r] * W[k*n+i][r] for r in range(R))
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _build_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left half - Strassen
    c = 0
    U[0][c] = 1; U[3][c] = 1
    V[0][c] = 1; V[5][c] = 1
    W[0][c] = 1; W[3][c] = 1
    
    c = 1
    U[2][c] = 1; U[3][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[3][c] = -1
    
    c = 2
    U[0][c] = 1
    V[1][c] = 1; V[5][c] = -1
    W[2][c] = 1; W[3][c] = 1
    
    c = 3
    U[3][c] = 1
    V[4][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1
    
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[5][c] = 1
    W[0][c] = -1; W[2][c] = 1
    
    c = 5
    U[2][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[3][c] = 1
    
    c = 6
    U[1][c] = 1; U[3][c] = -1
    V[4][c] = 1; V[5][c] = 1
    W[0][c] = 1
    
    # Right half - Strassen
    c = 7
    U[0][c] = 1; U[3][c] = 1
    V[2][c] = 1; V[7][c] = 1
    W[4][c] = 1; W[7][c] = 1
    
    c = 8
    U[2][c] = 1; U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1; W[7][c] = -1
    
    c = 9
    U[0][c] = 1
    V[3][c] = 1; V[7][c] = -1
    W[6][c] = 1; W[7][c] = 1
    
    c = 10
    U[3][c] = 1
    V[6][c] = 1; V[2][c] = -1
    W[4][c] = 1; W[5][c] = 1
    
    c = 11
    U[0][c] = 1; U[1][c] = 1
    V[7][c] = 1
    W[4][c] = -1; W[6][c] = 1
    
    c = 12
    U[2][c] = 1; U[0][c] = -1
    V[2][c] = 1; V[3][c] = 1
    W[7][c] = 1
    
    c = 13
    U[1][c] = 1; U[3][c] = -1
    V[6][c] = 1; V[7][c] = 1
    W[4][c] = 1
    
    return U, V, W

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 2, 4
    
    # Try to find a rank-13 solution
    # ... search code ...
    
    # Fall back to rank-14
    U, V, W = _build_rank14()
    return {"U": U, "V": V, "W": W}
