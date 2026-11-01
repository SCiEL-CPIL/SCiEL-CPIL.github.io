import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 factors
    # U2: 4x7 (rows: A11, A12, A21, A22)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1]
    ], dtype=int)
    
    # V2: 4x7 (rows: B11, B12, B21, B22)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    # W2: 4x7 (rows: C11, C12, C21, C22 in row-major order: C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Precompute permutation for 16 indices
    # k (kron index) = 8*i1 + 4*j1 + 2*i2 + j2
    # t (target index) = 8*i1 + 4*i2 + 2*j1 + j2
    perm = [0] * 16
    for k in range(16):
        i1 = k // 8
        rem = k % 8
        j1 = rem // 4
        rem2 = rem % 4
        i2 = rem2 // 2
        j2 = rem2 % 2
        t = 8 * i1 + 4 * i2 + 2 * j1 + j2
        perm[k] = t
        
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for a in range(7):
        for b in range(7):
            col = 7 * a + b
            
            # Compute Kronecker products
            u_vec = np.kron(U2[:, a], U2[:, b])
            v_vec = np.kron(V2[:, a], V2[:, b])
            w_vec = np.kron(W2[:, a], W2[:, b])
            
            # Apply permutation and fill columns
            for k in range(16):
                t = perm[k]
                U[t][col] = int(u_vec[k])
                V[t][col] = int(v_vec[k])
                W[t][col] = int(w_vec[k])
                
    return {"U": U, "V": V, "W": W}
