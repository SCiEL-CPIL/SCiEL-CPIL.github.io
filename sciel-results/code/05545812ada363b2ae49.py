import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively using additive splitting.
    # Rank estimate: 108
    
    def get_decomp(n, m, p):
        # Base case: Schoolbook
        if n == 1 or m == 1 or p == 1:
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
        
        # Base case: Strassen 2x2x2
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Indices:
            # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
            # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
            # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
            # W index: k*n + i.
            # C00: k=0,i=0 -> 0
            # C01: k=1,i=0 -> 1
            # C10: k=0,i=1 -> 2
            # C11: k=1,i=1 -> 3
            
            rank = 7
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1
            
            # M2
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = -1
            
            # M3
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[2][2] = 1
            
            # M4
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1
            
            # M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1
            
            # M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[3][5] = 1
            
            # M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[0][6] = 1; W[1][6] = -1
            
            return U, V, W

        # Recursive splitting
        # Choose split
        if n >= 2:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            
            # Combine
            # U1 is (n1*m, R1), U2 is (n2*m, R2)
            # We need to map indices.
            # For U2, the row indices are offset by n1*m.
            # For W, the indices are k*n + i.
            # For the second part, i is offset by n1.
            # So W2 indices need to be remapped.
            
            R1 = len(U1[0])
            R2 = len(U2[0])
            R = R1 + R2
            
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            # Copy U1
            for i in range(n1 * m):
                for r in range(R1):
                    U[i][r] = U1[i][r]
            
            # Copy U2 (offset rows)
            for i in range(n2 * m):
                for r in range(R2):
                    U[n1 * m + i][r + R1] = U2[i][r]
            
            # Copy V
            for i in range(m * p):
                for r in range(R1):
                    V[i][r] = V1[i][r]
                for r in range(R2):
                    V[i][r + R1] = V2[i][r]
            
            # Copy W (remap indices)
            # W1 indices: k*n + i for i in 0..n1-1
            # W2 indices: k*n + (i+n1) for i in 0..n2-1
            for k in range(p):
                for i in range(n1):
                    idx1 = k * n + i
                    for r in range(R1):
                        W[idx1][r] = W1[idx1][r]
                
                for i in range(n2):
                    idx2 = k * n + (i + n1)
                    # In W2, the index was k*n2 + i? No.
                    # W2 was constructed for dimensions n2, m, p.
                    # Its W indices are k*n2 + i.
                    # We need to map k*n2 + i to k*n + (i+n1).
                    idx_w2 = k * n2 + i
                    for r in range(R2):
                        W[idx2][r + R1] = W2[idx_w2][r]
                        
            return U, V, W

        elif m >= 2:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            
            R1 = len(U1[0])
            R2 = len(U2[0])
            R = R1 + R2
            
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            # U1: rows 0..n*m1-1
            for i in range(n * m1):
                for r in range(R1):
                    U[i][r] = U1[i][r]
            
            # U2: rows n*m1 .. n*m-1
            # In U2, row index is i (0..n*m2-1).
            # In U, row index is n*m1 + i.
            for i in range(n * m2):
                for r in range(R2):
                    U[n * m1 + i][r + R1] = U2[i][r]
            
            # V1: rows 0..m1*p-1
            for i in range(m1 * p):
                for r in range(R1):
                    V[i][r] = V1[i][r]
            
            # V2: rows m1*p .. m*p-1
            for i in range(m2 * p):
                for r in range(R2):
                    V[m1 * p + i][r + R1] = V2[i][r]
            
            # W: indices unchanged (k*n + i)
            for i in range(p * n):
                for r in range(R1):
                    W[i][r] = W1[i][r]
                for r in range(R2):
                    W[i][r + R1] = W2[i][r]
            
            return U, V, W

        elif p >= 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            
            R1 = len(U1[0])
            R2 = len(U2[0])
            R = R1 + R2
            
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            # U: unchanged
            for i in range(n * m):
                for r in range(R1):
                    U[i][r] = U1[i][r]
                for r in range(R2):
                    U[i][r + R1] = U2[i][r]
            
            # V1: rows 0..m*p1-1
            for i in range(m * p1):
                for r in range(R1):
                    V[i][r] = V1[i][r]
            
            # V2: rows m*p1 .. m*p-1
            for i in range(m * p2):
                for r in range(R2):
                    V[m * p1 + i][r + R1] = V2[i][r]
            
            # W1: indices k*n + i for k in 0..p1-1
            for k in range(p1):
                for i in range(n):
                    idx1 = k * n + i
                    for r in range(R1):
                        W[idx1][r] = W1[idx1][r]
            
            # W2: indices k*n + i for k in 0..p2-1
            # In W, index is (k+p1)*n + i
            for k in range(p2):
                for i in range(n):
                    idx2 = (k + p1) * n + i
                    idx_w2 = k * n + i
                    for r in range(R2):
                        W[idx2][r + R1] = W2[idx_w2][r]
            
            return U, V, W

    U, V, W = get_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}
