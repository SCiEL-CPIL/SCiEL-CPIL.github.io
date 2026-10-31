import numpy as np

def construct():
    # 2x2 Strassen matrices
    # U2[i*2+j, t]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Kronecker product gives the tensor square
    # Shape (16, 49)
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)

    # Permute rows to match 4x4 row-major indexing
    # Kron row index: k = k1*4 + k2
    # k1 = I*2 + J
    # k2 = i2*2 + j2
    # 4x4 row index: r = (2*I + i2)*4 + (2*J + j2)
    
    perm = np.zeros(16, dtype=int)
    for k1 in range(4):
        for k2 in range(4):
            I = k1 // 2
            J = k1 % 2
            i2 = k2 // 2
            j2 = k2 % 2
            
            r = (2*I + i2)*4 + (2*J + j2)
            perm[k1*4 + k2] = r

    # Apply permutation
    # U_kron[kron_row, col] moves to U[r, col]
    U = U_kron[perm, :]
    V = V_kron[perm, :]
    W = W_kron[perm, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
