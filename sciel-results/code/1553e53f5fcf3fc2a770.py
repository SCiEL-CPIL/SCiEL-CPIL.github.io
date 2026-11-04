import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2."""
    U = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # i=0, m=0
        [0,  1,  0,  0,  0,  0,  0],  # i=0, m=1
        [0,  0,  1,  0,  1,  0,  0],  # i=1, m=0
        [0,  0,  1,  0,  0,  1,  0],  # i=1, m=1
    ], dtype=float)
    V = np.array([
        [1,  1,  0,  0,  0,  0,  0],  # m=0, k=0
        [0,  0,  1,  1,  0,  0,  0],  # m=0, k=1
        [1, -1,  0,  0,  0,  0,  0],  # m=1, k=0
        [0,  0,  1, -1,  0,  0,  0],  # m=1, k=1
    ], dtype=float)
    W = np.array([
        [0,  0,  1,  0,  0,  0,  0],  # k=0, i=0
        [1,  1,  0,  0,  0,  0,  0],  # k=0, i=1
        [0,  0,  0,  0,  1,  1,  1],  # k=1, i=0
        [0,  0,  0,  0,  0,  1,  0],  # k=1, i=1
    ], dtype=float)
    return U, V, W

def _schoolbook(n, m, p):
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

def _block_decompose(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    if n * m * p <= 4:
        return _schoolbook(n, m, p)

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try splitting middle dimension
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _block_decompose(n, m1, p)
        U2, V2, W2 = _block_decompose(n, m2, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        if total_rank < best_rank:
            best_rank = total_rank
            U = np.zeros((n * m, total_rank))
            V = np.zeros((m * p, total_rank))
            W = np.zeros((p * n, total_rank))
            
            # Left part (j in 0..m1-1)
            for i in range(n):
                for j in range(m1):
                    U[i * m + j, :r1] = U1[i * m1 + j, :]
            for j in range(m1):
                for k in range(p):
                    V[j * p + k, :r1] = V1[j * p + k, :]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, :r1] = W1[k * n + i, :]
                    
            # Right part (j in m1..m-1)
            for i in range(n):
                for j in range(m2):
                    U[i * m + m1 + j, r1:] = U2[i * m2 + j, :]
            for j in range(m2):
                for k in range(p):
                    V[(m1 + j) * p + k, r1:] = V2[j * p + k, :]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r1:] = W2[k * n + i, :]
            
            best_U, best_V, best_W = U, V, W
            
    if best_U is not None:
        return best_U, best_V, best_W
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _block_decompose(n, m, p)
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
