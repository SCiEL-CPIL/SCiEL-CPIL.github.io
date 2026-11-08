import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    # The naive tensor rank for (2,4,8) matrix multiplication is m=4.
    # The tensor is T_{i,j,k} = delta_{i,j,k} where C[i,k] = sum_j A[i,j]B[j,k]
    # This is a rank-4 tensor (the "naive" decomposition).
    # But we want rank-minimal EXACT decomposition.
    # 
    # For matrix multiplication M_{n,m,p}, the tensor has rank at most m (naive).
    # The question is whether we can do better.
    #
    # For 2x4x8, the naive rank is 4. Can we do rank 3?
    # The border rank and tensor rank for small matrix multiplication are known.
    # For 2x2x2, rank is 7 (Strassen) but naive is 2. Wait no - naive for 2x2x2 is 2.
    # Actually for M_{n,m,p}, the naive decomposition has rank m.
    # 
    # Wait, I need to reconsider. The tensor for matrix multiplication M_{n,m,p}
    # has entries T_{(i,j),(j,k),(k,i)} = 1 if the indices match, else 0.
    # The naive rank is m (one term per intermediate index j).
    # 
    # For M_{2,4,8}: naive rank = 4.
    # Is there a rank-3 decomposition? Unlikely for these dimensions.
    # 
    # The current program uses Strassen blocks giving rank 56, which is much worse than 4.
    # The best we can do is rank 4 (the naive decomposition).
    #
    # Let's construct the naive rank-4 decomposition.
    # A is 2x4, B is 4x8, C = A*B is 2x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # In the trace convention: W[k*n+i, r]
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (16, rank)
    #
    # For rank r = 4 (one term per j):
    # Term j (j=0,1,2,3):
    #   U: A[i,j] -> U[i*m+j, j] = 1
    #   V: B[j,k] -> V[j*p+k, j] = 1
    #   W: C[k,i] -> W[k*n+i, j] = 1

    rank = m  # = 4

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
