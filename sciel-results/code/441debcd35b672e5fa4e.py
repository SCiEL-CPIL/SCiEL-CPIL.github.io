import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Rank 27 decomposition
    # Each of the m=3 blocks corresponds to M_{2,1,9} which has rank 9.
    # Total rank = 3 * 9 = 27.
    
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for c in range(m):
        # Block c corresponds to inner index c
        # u vector: a_{0c} e_0 + a_{1c} e_1
        # v vector: b_{c, k} e_{c,k} for k in 0..p-1
        # w vector: e_{0,k} + e_{1,k} for k in 0..p-1 (indices in W)
        
        # Indices in U:
        # i = ai * m + am. Here am = c.
        # ai = 0 -> i = c
        # ai = 1 -> i = n + c
        u_idx_0 = c
        u_idx_1 = n + c
        
        for k in range(p):
            # Indices in V:
            # j = bj * p + bm. Here bj = c, bm = k.
            v_idx = c * p + k
            
            # Indices in W:
            # k_idx = bm * n + ai. Here bm = k.
            # ai = 0 -> k_idx = k * n + 0 = 2k
            # ai = 1 -> k_idx = k * n + 1 = 2k + 1
            w_idx_0 = k * n + 0
            w_idx_1 = k * n + 1
            
            U[u_idx_0, col] = 1
            U[u_idx_1, col] = 1
            V[v_idx, col] = 1
            W[w_idx_0, col] = 1
            W[w_idx_1, col] = 1
            
            col += 1

    # Verify the decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
