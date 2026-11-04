import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _build_strassen_2x2(i1, i2, j1, j2, k1, k2, U, V, W, r):
    """
    Adds a Strassen 2x2x2 decomposition to U, V, W starting at column r.
    Returns the new r (r+7).
    Indices:
    i1, i2: row indices in A (2 rows)
    j1, j2: col indices in A / row indices in B (2 cols/rows)
    k1, k2: col indices in B (2 cols)
    """
    # M1: (A11 + A22)(B11 + B22)
    U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
    V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
    W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
    r += 1
    
    # M2: (A21 + A22)B11
    U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
    V[j1*p+k1][r] = 1
    W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
    r += 1
    
    # M3: A11(B21 - B11)
    U[i1*m+j1][r] = 1
    V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
    W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
    r += 1
    
    # M4: A22(B12 - B21)
    U[i2*m+j2][r] = 1
    V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
    W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
    r += 1
    
    # M5: (A11 + A12)B22
    U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
    V[j2*p+k2][r] = 1
    W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
    r += 1
    
    # M6: (A21 - A11)(B11 + B12)
    U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
    V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
    W[k2*n+i2][r] = 1
    r += 1
    
    # M7: (A12 - A22)(B21 + B22)
    U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
    V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
    W[k1*n+i1][r] = 1
    r += 1
    
    return r

def _build_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2 (Column 2 of A, Row 2 of B)
    # Direct multiplication: 2x1 * 1x5 = 2x5
    # 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1 and k=4
    # Direct multiplication: 2x2 * 2x1 = 2x2
    # 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1 and k=0,1
    # Strassen 2x2x2: 7 terms
    r = _build_strassen_2x2(0, 1, 0, 1, 0, 1, U, V, W, r)
    
    # Part 4: j=0,1 and k=2,3
    # Strassen 2x2x2: 7 terms
    r = _build_strassen_2x2(0, 1, 0, 1, 2, 3, U, V, W, r)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for ri in range(rank):
                    s += U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
            
    if ok:
        return {"U": U, "V": V, "W": W}
    else:
        return None

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Attempt Rank 28
    result = _build_rank_28()
    if result is not None:
        return result
        
    # Fallback to Schoolbook (Rank 30)
    return _build_schoolbook()
