# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    n, m, p = 2, 2, 16
    rank = n * m * p  # 64

    # Indices
    i = np.arange(n)
    j = np.arange(m)
    k = np.arange(p)
    
    # Create meshgrid for all combinations
    # We need to map (i, j, k) to column index
    # col = i * m * p + j * p + k
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i

    # Generate all combinations
    # Using np.indices or broadcasting
    I, J, K = np.meshgrid(i, j, k, indexing='ij')
    I = I.flatten()
    J = J.flatten()
    K = K.flatten()
    
    cols = I * m * p + J * p + K
    u_idx = I * m + J
    v_idx = J * p + K
    w_idx = K * n + I

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    U[u_idx, cols] = 1
    V[v_idx, cols] = 1
    W[w_idx, cols] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
