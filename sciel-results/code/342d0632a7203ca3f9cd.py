import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = 4, 4, 9  # A is n×m, B is m×p, C is n×p
    
    # Trivial exact decomposition: rank = n*m*p = 4*4*9 = 144
    # Each rank-1 term corresponds to one (i,k,j) triple where i∈[0,n), k∈[0,m), j∈[0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    rank = n * m * p  # 144
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U row: i*m + k (A[i,k])
                # V row: k*p + j (B[k,j])
                # W row: j*n + i (C[i,j]) - using column-major style indexing
                U[i * m + k, idx] = 1
                V[k * p + j, idx] = 1
                W[j * n + i, idx] = 1
                idx += 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        pass  # We already have our answer, just safety check
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
