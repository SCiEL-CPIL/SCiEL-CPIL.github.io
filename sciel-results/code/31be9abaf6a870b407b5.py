import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Exact check for tensor decomposition of matrix multiplication.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    """
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block():
    """
    Returns U, V, W for the 2x2x2 Strassen block.
    U, V, W are 4x7 matrices (rows are indices, cols are terms).
    Indices:
    U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # U: 00, 11 -> 1. V: 00, 11 -> 1. W: 00, 11 -> 1? 
    # Strassen terms are not simply diagonal.
    # We use the known 7-term decomposition.
    # Term 0: M1
    # Term 1: M2
    # ...
    
    # Coefficients from standard Strassen 2x2 decomposition of the tensor.
    # This is a known solution.
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _hybrid_decomposition(n, m, p):
    """
    Combines Strassen 2x2 block with Schoolbook for the rest.
    Rank = 7 + (n*m*p - 8)
    """
    rank = 7 + (n * m * p - 8)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Place Strassen block
    sU, sV, sW = _strassen_2x2_block()
    for i in range(4):
        for c in range(7):
            # Map local indices to global
            # Local i: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            li = i // 2
            lj = i % 2
            U[li * m + lj][c] = sU[i][c]
            
            lv = i // 2
            lk = i % 2
            V[lv * p + lk][c] = sV[i][c]
            
            lw = i // 2
            li_w = i % 2
            W[lw * n + li_w][c] = sW[i][c]

    # 2. Place Schoolbook terms for the rest
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try hybrid (Rank 44)
    res = _hybrid_decomposition(n, m, p)
    if _validate(res["U"], res["V"], res["W"], n, m, p):
        return res
        
    # Fallback to Schoolbook (Rank 45)
    return _schoolbook(n, m, p)
