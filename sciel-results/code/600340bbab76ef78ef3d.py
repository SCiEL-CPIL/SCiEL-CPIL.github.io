import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-minimal decomposition for M_{n,m,p} = M_{n,m,1} ⊗ I_p
    # Rank(M_{n,m,1}) = n*m
    # Total rank = n*m*p
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Schoolbook / bilinear decomposition
    for k in range(p):
        for i in range(n):
            for j in range(m):
                col = k * (n * m) + i * m + j
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
