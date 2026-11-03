import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Indices:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    W index: k*n + i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    # C10 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10 -> k=0, i=1 -> idx 1
    
    # M3 = A00(B01 - B11)
    # C01 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1 # C01 -> k=1, i=0 -> idx 2
    
    # M4 = A11(B10 - B00)
    # C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00 + A01)B11
    # C01 += M5, C00 -= M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # C00 += M7, C11 -= M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return {"U": U, "V": V, "W": W}

def _combine(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    """
    Combine two decompositions for n1xm x p and n2xm x p into (n1+n2)xm x p.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    n = n1 + n2
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r1):
        for i in range(n1):
            for j in range(m):
                U[i*m+j][c] = U1[i*m+j][c]
        for j in range(m):
            for k in range(p):
                V[j*p+k][c] = V1[j*p+k][c]
        for k in range(p):
            for i in range(n1):
                W[k*n1+i][c] = W1[k*n1+i][c] # Note: W index in W1 is k*n1+i
        
    for c in range(r2):
        for i in range(n2):
            for j in range(m):
                U[(n1+i)*m+j][c+r1] = U2[i*m+j][c]
        for j in range(m):
            for k in range(p):
                V[j*p+k][c+r1] = V2[j*p+k][c]
        for k in range(p):
            for i in range(n2):
                W[k*n+(n1+i)][c+r1] = W2[k*n2+i][c]
                
    return {"U": U, "V": V, "W": W}

def _combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two decompositions for nxm1xp and nxm2xp into nx(m1+m2)xp.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    m = m1 + m2
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r1):
        for i in range(n):
            for j in range(m1):
                U[i*m+j][c] = U1[i*m1+j][c]
        for j in range(m1):
            for k in range(p):
                V[j*p+k][c] = V1[j*p+k][c]
        for k in range(p):
            for i in range(n):
                W[k*n+i][c] = W1[k*n+i][c]
        
    for c in range(r2):
        for i in range(n):
            for j in range(m2):
                U[i*m+(m1+j)][c+r1] = U2[i*m2+j][c]
        for j in range(m2):
            for k in range(p):
                V[(m1+j)*p+k][c+r1] = V2[j*p+k][c]
        for k in range(p):
            for i in range(n):
                W[k*n+i][c+r1] = W2[k*n+i][c]
                
    return {"U": U, "V": V, "W": W}

def _combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two decompositions for nxmxp1 and nxmxp2 into nxmx(p1+p2).
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    p = p1 + p2
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r1):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = U1[i*m+j][c]
        for j in range(m):
            for k in range(p1):
                V[j*p+k][c] = V1[j*p1+k][c]
        for k in range(p1):
            for i in range(n):
                W[k*n+i][c] = W1[k*n+i][c]
        
    for c in range(r2):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c+r1] = U2[i*m+j][c]
        for j in range(m):
            for k in range(p2):
                V[j*p+(p1+k)][c+r1] = V2[j*p2+k][c]
        for k in range(p2):
            for i in range(n):
                W[(p1+k)*n+i][c+r1] = W2[k*n+i][c]
                
    return {"U": U, "V": V, "W": W}

def _decompose(n, m, p):
    if n == 1 or m == 1 or p == 1:
        rank = 1
        U = [[1]*rank for _ in range(n*m)]
        V = [[1]*rank for _ in range(m*p)]
        W = [[1]*rank for _ in range(p*n)]
        return {"U": U, "V": V, "W": W}
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    if n == 3 and m == 3 and p == 3:
        # Placeholder for rank 23, but we don't have it. 
        # We will rely on splitting.
        pass
        
    best_rank = float('inf')
    best_result = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        res1 = _decompose(n1, m, p)
        res2 = _decompose(n2, m, p)
        r = len(res1["U"][0]) + len(res2["U"][0])
        if r < best_rank:
            best_rank = r
            best_result = _combine(res1["U"], res1["V"], res1["W"], res2["U"], res2["V"], res2["W"], n1, n2, m, p)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        res1 = _decompose(n, m1, p)
        res2 = _decompose(n, m2, p)
        r = len(res1["U"][0]) + len(res2["U"][0])
        if r < best_rank:
            best_rank = r
            best_result = _combine_m(res1["U"], res1["V"], res1["W"], res2["U"], res2["V"], res2["W"], n, m1, m2, p)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        res1 = _decompose(n, m, p1)
        res2 = _decompose(n, m, p2)
        r = len(res1["U"][0]) + len(res2["U"][0])
        if r < best_rank:
            best_rank = r
            best_result = _combine_p(res1["U"], res1["V"], res1["W"], res2["U"], res2["V"], res2["W"], n, m, p1, p2)
            
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We can just call _decompose
    result = _decompose(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        return _build_schoolbook(n, m, p)

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
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
