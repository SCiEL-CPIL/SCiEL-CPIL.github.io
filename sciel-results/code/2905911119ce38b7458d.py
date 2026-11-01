import numpy as np

def construct():
    n, m, p = 2, 3, 5
    
    # Decompose into (2,1,1) and (1,3,5)
    n1, m1, p1 = 2, 1, 1
    n2, m2, p2 = 1, 3, 5
    
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U1, V1, W1 = get_schoolbook(n1, m1, p1)
    U2, V2, W2 = get_schoolbook(n2, m2, p2)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    U_kron = np.kron(U1, U2)
    V_kron = np.kron(V1, V2)
    W_kron = np.kron(W1, W2)
    
    # Permute rows to match standard indexing
    # U index: i*m + j
    # Kron U index: (i1*m1 + j1)*(n2*m2) + (i2*m2 + j2)
    # Target U index: (i1*n2 + i2)*(m1*m2) + (j1*m2 + j2)
    
    perm_U = np.zeros(n*m, dtype=int)
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    idx_kron = (i1 * m1 + j1) * (n2 * m2) + (i2 * m2 + j2)
                    idx_target = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                    perm_U[idx_target] = idx_kron
                    
    U = U_kron[perm_U, :]
    
    # V index: j*p + k
    # Kron V index: (j1*p1 + k1)*(m2*p2) + (j2*p2 + k2)
    # Target V index: (j1*m2 + j2)*(p1*p2) + (k1*p2 + k2)
    
    perm_V = np.zeros(m*p, dtype=int)
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    idx_kron = (j1 * p1 + k1) * (m2 * p2) + (j2 * p2 + k2)
                    idx_target = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                    perm_V[idx_target] = idx_kron
                    
    V = V_kron[perm_V, :]
    
    # W index: k*n + i
    # Kron W index: (k1*n1 + i1)*(p2*n2) + (k2*n2 + i2)
    # Target W index: (k1*n2 + k2)*(n1*n2) + (i1*n2 + i2)
    
    perm_W = np.zeros(p*n, dtype=int)
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    idx_kron = (k1 * n1 + i1) * (p2 * n2) + (k2 * n2 + i2)
                    idx_target = (k1 * n2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                    perm_W[idx_target] = idx_kron
                    
    W = W_kron[perm_W, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
