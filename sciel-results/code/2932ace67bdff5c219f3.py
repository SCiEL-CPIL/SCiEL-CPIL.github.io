import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Define 2x2 Strassen matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    
    # U2 rows: A00, A01, A10, A11
    # V2 rows: B00, B01, B10, B11
    # W2 rows: C00, C10, C01, C11 (Convention W[k*n+i, c] with n=2:
    # k=0,i=0 -> 0 (C00); k=0,i=1 -> 1 (C10); k=1,i=0 -> 2 (C01); k=1,i=1 -> 3 (C11))
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=int)
    
    # Kronecker product for 4x4
    # U4 shape: (4*4, 7*7) = (16, 49)
    # V4 shape: (4*4, 7*7) = (16, 49)
    # W4 shape: (4*4, 7*7) = (16, 49)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Check time, though this is very fast
    if time.time() - start_time > 110:
        # Fallback to schoolbook if somehow this takes too long (unlikely)
        n, m, p = 4, 4, 4
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
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
