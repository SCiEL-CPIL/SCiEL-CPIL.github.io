"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
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
    return U, V, W

def _outer_product(n, m, p):
    # Handles cases where one dimension is 1
    if m == 1:
        # A is n x 1, B is 1 x p. C is n x p. C[i,k] = A[i] * B[k]
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * m][col] = 1
                V[k][col] = 1 # j is 0
                W[k * n + i][col] = 1
                col += 1
        return U, V, W
    elif n == 1:
        # A is 1 x m, B is m x p. C is 1 x p. C[0,k] = sum A[j] B[j,k]
        # This is not an outer product, it's a sum.
        # But if n=1, it's just vector-matrix mult.
        # Schoolbook is m*p. Can we do better?
        # If m=1, handled above.
        # If p=1, handled below.
        pass
    elif p == 1:
        # A is n x m, B is m x 1. C is n x 1. C[i] = sum A[i,j] B[j]
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1 # k is 0
                W[i][col] = 1 # k*n + i = i
                col += 1
        return U, V, W
    # Fallback to schoolbook for other 1-dim cases if not simple outer
    return _schoolbook(n, m, p)

def _strassen_222():
    # Returns U, V, W for M_2,2,2
    # Indices: A(2x2), B(2x2), C(2x2)
    # U rows: 0..3 (i*m+j)
    # V rows: 0..3 (j*p+k)
    # W rows: 0..3 (k*n+i)
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Col 0: M1
    U[0][0] = 1; U[3][0] = 1 # A11, A22
    V[0][0] = 1; V[3][0] = 1 # B11, B22 (j*p+k: 0*2+0=0, 1*2+1=3)
    W[0][0] = 1; W[3][0] = 1 # C11, C22 (k*n+i: 0*2+0=0, 1*2+1=3)
    
    # Col 1: M2
    U[2][1] = 1; U[3][1] = 1 # A21, A22
    V[0][1] = 1 # B11
    W[2][1] = 1; W[3][1] = 1 # C21, C22
    
    # Col 2: M3
    U[0][2] = 1 # A11
    V[1][2] = 1; V[3][2] = -1 # B12, B22
    W[1][2] = 1; W[3][2] = 1 # C12, C22
    
    # Col 3: M4
    U[3][3] = 1 # A22
    V[2][3] = 1; V[0][3] = -1 # B21, B11
    W[0][3] = 1; W[2][3] = 1 # C11, C21
    
    # Col 4: M5
    U[0][4] = 1; U[1][4] = 1 # A11, A12
    V[3][4] = 1 # B22
    W[0][4] = -1; W[1][4] = 1 # C11, C12
    
    # Col 5: M6
    U[2][5] = 1; U[0][5] = -1 # A21, A11
    V[0][5] = 1; V[1][5] = 1 # B11, B12
    W[0][5] = 1; W[3][5] = 1 # C11, C22
    
    # Col 6: M7
    U[1][6] = 1; U[3][6] = -1 # A12, A22
    V[2][6] = 1; V[3][6] = 1 # B21, B22
    W[0][6] = 1; W[1][6] = 1 # C11, C12
    
    return U, V, W

def _combine_inner_split(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
    # A = [A1 | A2], B = [B1; B2]
    # C = A1 B1 + A2 B2
    # U1 shape (n*m1, r1), U2 shape (n*m2, r2)
    # U shape (n*(m1+m2), r1+r2)
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 + r2
    U = [[0]*rank for _ in range(n*(m1+m2))]
    V = [[0]*rank for _ in range((m1+m2)*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill U
    # Rows 0..n*m1-1 correspond to A1
    for i in range(n*m1):
        for c in range(r1):
            U[i][c] = U1[i][c]
    # Rows n*m1..n*(m1+m2)-1 correspond to A2
    for i in range(n*m2):
        for c in range(r2):
            U[n*m1 + i][r1 + c] = U2[i][c]
            
    # Fill V
    # Rows 0..m1*p-1 correspond to B1
    for j in range(m1*p):
        for c in range(r1):
            V[j][c] = V1[j][c]
    # Rows m1*p..(m1+m2)*p-1 correspond to B2
    for j in range(m2*p):
        for c in range(r2):
            V[m1*p + j][r1 + c] = V2[j][c]
            
    # Fill W
    for k in range(p*n):
        for c in range(r1):
            W[k][c] = W1[k][c]
        for c in range(r2):
            W[k][r1 + c] = W2[k][c]
            
    return U, V, W

def _combine_outer_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    # C = [C1 | C2]
    # C1 = A B1, C2 = A B2
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 + r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*(p1+p2))]
    W = [[0]*rank for _ in range((p1+p2)*n)]
    
    # U is same for both
    for i in range(n*m):
        for c in range(r1):
            U[i][c] = U1[i][c]
        for c in range(r2):
            U[i][r1 + c] = U2[i][c]
            
    # V
    for j in range(m*p1):
        for c in range(r1):
            V[j][c] = V1[j][c]
    for j in range(m*p2):
        for c in range(r2):
            V[m*p1 + j][r1 + c] = V2[j][c]
            
    # W
    for k in range(p1*n):
        for c in range(r1):
            W[k][c] = W1[k][c]
    for k in range(p2*n):
        for c in range(r2):
            W[p1*n + k][r1 + c] = W2[k][c]
            
    return U, V, W

def _get_decomp(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    if m == 1 or n == 1 or p == 1:
        return _outer_product(n, m, p)
        
    # Heuristic: Split the dimension that allows best recursion
    # Try splitting p
    if p > 1:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _get_decomp(n, m, p1)
        U2, V2, W2 = _get_decomp(n, m, p2)
        return _combine_outer_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2)
    # Try splitting m
    if m > 1:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _get_decomp(n, m1, p)
        U2, V2, W2 = _get_decomp(n, m2, p)
        return _combine_inner_split(n, m1, m2, p, U1, V1, W1, U2, V2, W2)
    # Try splitting n
    if n > 1:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _get_decomp(n1, m, p)
        U2, V2, W2 = _get_decomp(n2, m, p)
        # Combine n split
        r1 = len(W1[0])
        r2 = len(W2[0])
        rank = r1 + r2
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for i in range(n1*m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for i in range(n2*m):
            for c in range(r2):
                U[n1*m + i][r1 + c] = U2[i][c]
                
        for j in range(m*p):
            for c in range(r1):
                V[j][c] = V1[j][c]
            for c in range(r2):
                V[j][r1 + c] = V2[j][c]
                
        for k in range(p*n1):
            for c in range(r1):
                W[k][c] = W1[k][c]
        for k in range(p*n2):
            for c in range(r2):
                W[p*n1 + k][r1 + c] = W2[k][c]
        return U, V, W
        
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
