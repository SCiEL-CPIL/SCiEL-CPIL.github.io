import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    
    # For n=2, m=5, p=5:
    # The matrix multiplication C = A * B where A is 2x5 and B is 5x5
    # can be decomposed into two independent 1x5x5 multiplications:
    # Row 0 of C = Row 0 of A * B
    # Row 1 of C = Row 1 of A * B
    #
    # Each 1x5x5 multiplication has rank 5 (it's just a row vector times a matrix).
    # For 1x5x5:
    #   T_{i,j,k} = delta_{i,0} * 1 (for the single row)
    #   The rank-5 decomposition:
    #     For r = 0..4:
    #       U_{j, r} = delta_{j,r}  (selects A_j)
    #       V_{j*p+k, r} = 1 if j=r  (selects B_{r,k})
    #       W_{k*n+i, r} = 1 if i=0  (adds to C_k)
    #
    # For n=2, m=5, p=5:
    #   U shape: (n*m, rank) = (10, 10)
    #   V shape: (m*p, rank) = (25, 10)
    #   W shape: (p*n, rank) = (10, 10)
    #
    # Decomposition:
    #   For i=0 (first row of A), r ranges 0..4:
    #     U[i*m+j, r] = 1 if j=r
    #     V[j*p+k, r] = 1 if j=r
    #     W[k*n+i, r] = 1 if i=0
    #
    #   For i=1 (second row of A), r ranges 5..9, let r' = r-5:
    #     U[i*m+j, r] = 1 if j=r'
    #     V[j*p+k, r] = 1 if j=r'
    #     W[k*n+i, r] = 1 if i=1
    
    rank = 10
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # First row of A (i=0), terms r=0..4
    for r in range(5):
        for j in range(m):
            if j == r:
                U[0 * m + j, r] = 1
        for j in range(m):
            if j == r:
                for k in range(p):
                    V[j * p + k, r] = 1
        for k in range(p):
            W[k * n + 0, r] = 1
    
    # Second row of A (i=1), terms r=5..9, r' = r-5
    for r in range(5, 10):
        r_prime = r - 5
        for j in range(m):
            if j == r_prime:
                U[1 * m + j, r] = 1
        for j in range(m):
            if j == r_prime:
                for k in range(p):
                    V[j * p + k, r] = 1
        for k in range(p):
            W[k * n + 1, r] = 1
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
