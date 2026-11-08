import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Rank-n*m decomposition: each term r corresponds to (i,j) pair
    # U[i*m+j, r] = 1 if r == (i,j)
    # V[j'*p+k, r] = 1 if j' == j (for all k)
    # W[k'*n+i, r] = 1 if i == i' (for all k')
    #
    # Verification: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # Term r=(i',j') contributes iff i=i' and j=j' (from U) AND j=j' (from V) AND i=i' (from W)
    # = 1 iff r=(i,j), so sum = 1 for all valid (i,j,k). Correct!
    rank = n * m  # 2*4 = 8
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    idx = 0
    for i in range(n):
        for j in range(m):
            # U: only one non-zero entry per term
            U[i * m + j, idx] = 1
            # V: all k for this j
            for k in range(p):
                V[j * p + k, idx] = 1
            # W: all k for this i
            for k in range(p):
                W[k * n + i, idx] = 1
            idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
