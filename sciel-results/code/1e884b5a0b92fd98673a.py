import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 82
    # Block 1: A[0:3, 0:3] B[0:3, 0:3] -> C[0:3, 0:3] (Rank 23)
    # Block 2: A[0:3, 3:5] B[3:5, 0:3] -> C[0:3, 0:3] (Rank 18)
    # Block 3: A[0:3, 0:3] B[0:3, 3:6] -> C[0:3, 3:6] (Rank 23)
    # Block 4: A[0:3, 3:5] B[3:5, 3:6] -> C[0:3, 3:6] (Rank 18)
    
    m3_terms = [
        ((1,0,0), (1,0,0), (1,0,0)),
        ((0,1,0), (1,0,0), (0,0,1)),
        ((0,0,1), (1,0,0), (0,1,0)),
        ((1,0,0), (0,1,0), (0,1,0)),
        ((0,1,0), (0,1,0), (0,0,1)),
        ((0,0,1), (0,1,0), (0,0,1)),
        ((1,0,0), (0,0,1), (0,1,0)),
        ((0,1,0), (0,0,1), (0,0,1)),
        ((0,0,1), (0,0,1), (0,0,1)),
        ((1,1,0), (0,0,0), (0,0,1)),
        ((0,0,0), (1,1,0), (0,0,1)),
        ((0,0,1), (0,0,1), (1,0,0)),
        ((0,1,1), (0,0,0), (0,1,0)),
        ((1,0,1), (0,0,0), (0,1,0)),
        ((0,0,1), (1,0,1), (0,0,0)),
        ((0,0,0), (0,1,1), (0,1,0)),
        ((1,0,1), (0,0,0), (0,1,0)),
        ((0,0,1), (0,0,1), (1,0,0)),
        ((0,1,0), (1,0,0), (0,0,1)),
        ((0,0,1), (0,0,0), (1,1,0)),
        ((0,0,0), (0,1,0), (1,1,0)),
        ((1,0,0), (0,0,1), (0,1,0)),
        ((0,1,0), (0,0,0), (0,1,1))
    ]
    
    # 3x1x3 terms (Rank 9)
    m313_terms = []
    for i in range(3):
        for k in range(3):
            u = [0,0,0]; u[i] = 1
            v = [1] # scalar
            w = [0,0,0]; w[k] = 1
            m313_terms.append((u, v, w))
            
    rank = 23 + 18 + 23 + 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Block 1: A[0:3, 0:3] B[0:3, 0:3]
    for u, v, w in m3_terms:
        for i in range(3):
            for j in range(3):
                U[i*m + j][col] = u[i]
        for j in range(3):
            for k in range(3):
                V[j*p + k][col] = v[j]
        for k in range(3):
            for i in range(3):
                W[k*n + i][col] = w[k]
        col += 1
        
    # Block 2: A[0:3, 3:5] B[3:5, 0:3]
    # A global I in 0..2, J in 3..4. B global J in 3..4, K in 0..2. C I in 0..2, K in 0..2.
    for u, v, w in m313_terms:
        for i in range(3):
            for j in range(2):
                U[i*m + (3+j)][col] = u[i]
        for j in range(2):
            for k in range(3):
                V[(3+j)*p + k][col] = v[0]
        for k in range(3):
            for i in range(3):
                W[k*n + i][col] = w[k]
        col += 1
        
    # Block 3: A[0:3, 0:3] B[0:3, 3:6]
    for u, v, w in m3_terms:
        for i in range(3):
            for j in range(3):
                U[i*m + j][col] = u[i]
        for j in range(3):
            for k in range(3):
                V[j*p + (3+k)][col] = v[j]
        for k in range(3):
            for i in range(3):
                W[(3+k)*n + i][col] = w[k]
        col += 1
        
    # Block 4: A[0:3, 3:5] B[3:5, 3:6]
    for u, v, w in m313_terms:
        for i in range(3):
            for j in range(2):
                U[i*m + (3+j)][col] = u[i]
        for j in range(2):
            for k in range(3):
                V[(3+j)*p + (3+k)][col] = v[0]
        for k in range(3):
            for i in range(3):
                W[(3+k)*n + i][col] = w[k]
        col += 1
        
    return {"U": U, "V": V, "W": W}
