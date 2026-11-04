import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def decompose(n, m, p):
    # Returns U (n*m, R), V (m*p, R), W (p*n, R)
    
    # Base cases: Schoolbook
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
        
    if n == 1:
        # m x p
        rank = m * p
        U = [[0]*rank for _ in range(m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j*p+k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W
        
    if m == 1:
        # n x p
        rank = n * p
        U = [[0]*rank for _ in range(n)]
        V = [[0]*rank for _ in range(p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        return U, V, W
        
    if p == 1:
        # n x m
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m)]
        W = [[0]*rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Strassen 2x2x2
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Indices:
        # U rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # V rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        # W rows: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = -1
        W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Recursive Split
    # Try splitting n, m, p
    # For 4x5x6, we determined split m=5 into 2+3 is good.
    # We can hardcode the best splits or search.
    # Given small dims, search is fine.
    
    best_rank = n*m*p
    best_U, best_V, best_W = None, None, None
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = decompose(n1, m, p)
        U2, V2, W2 = decompose(n2, m, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        if r < best_rank:
            # Combine
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # U: top n1*m rows from U1, bottom n2*m rows from U2
            for i in range(n1*m):
                U[i] = U1[i] + [0]*r2
            for i in range(n2*m):
                U[n1*m+i] = [0]*r1 + U2[i]
                
            # V: V1 and V2 are for the same B, but different terms.
            # Actually, V1 is (m*p, r1), V2 is (m*p, r2).
            # We just concatenate columns.
            for j in range(m*p):
                V[j] = V1[j] + V2[j]
                
            # W: W1 is (p*n1, r1), W2 is (p*n2, r2).
            # W rows are indexed by k*n + i.
            # For W1, i < n1. For W2, i >= n1.
            for k in range(p):
                for i in range(n1):
                    W[k*n+i] = W1[k*n1+i] + [0]*r2
                for i in range(n2):
                    W[k*n+n1+i] = [0]*r1 + W2[k*n2+i]
                    
            best_rank = r
            best_U, best_V, best_W = U, V, W

    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = decompose(n, m1, p)
        U2, V2, W2 = decompose(n, m2, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        if r < best_rank:
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # U: rows i*m + j. j < m1 from U1, j >= m1 from U2.
            for i in range(n):
                for j in range(m1):
                    U[i*m+j] = U1[i*m1+j] + [0]*r2
                for j in range(m2):
                    U[i*m+m1+j] = [0]*r1 + U2[i*m2+j]
                    
            # V: rows j*p + k. j < m1 from V1, j >= m1 from V2.
            for j in range(m1):
                for k in range(p):
                    V[j*p+k] = V1[j*p+k] + [0]*r2
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p+k] = [0]*r1 + V2[j*p+k]
                    
            # W: rows k*n + i. Same for both.
            for k in range(p):
                for i in range(n):
                    W[k*n+i] = W1[k*n+i] + W2[k*n+i]
                    
            best_rank = r
            best_U, best_V, best_W = U, V, W

    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = decompose(n, m, p1)
        U2, V2, W2 = decompose(n, m, p2)
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        if r < best_rank:
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # U: Same for both.
            for i in range(n*m):
                U[i] = U1[i] + U2[i]
                
            # V: rows j*p + k. k < p1 from V1, k >= p1 from V2.
            for j in range(m):
                for k in range(p1):
                    V[j*p+k] = V1[j*p1+k] + [0]*r2
                for k in range(p2):
                    V[j*p+p1+k] = [0]*r1 + V2[j*p2+k]
                    
            # W: rows k*n + i. k < p1 from W1, k >= p1 from W2.
            for k in range(p1):
                for i in range(n):
                    W[k*n+i] = W1[k*n1+i] + [0]*r2 # Wait, W1 shape is p1*n, r1.
            # Correction on W indices for p-split
            for k in range(p1):
                for i in range(n):
                    W[k*n+i] = W1[k*n+i] + [0]*r2
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n+i] = [0]*r1 + W2[k*n+i]
                    
            best_rank = r
            best_U, best_V, best_W = U, V, W

    return best_U, best_V, best_W
