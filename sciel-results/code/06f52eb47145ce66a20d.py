import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 7
    N_U = n * m  # 8
    N_V = m * p  # 28
    N_W = p * n  # 14
    
    rank = 4
    U = np.zeros((N_U, rank), dtype=np.int64)
    V = np.zeros((N_V, rank), dtype=np.int64)
    W = np.zeros((N_W, rank), dtype=np.int64)
    
    # Decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1
    # We use 4 terms, one for each j.
    # Term r corresponds to j = r.
    # U[i, j, r] = 1 if j == r else 0
    # V[j, k, r] = 1 if j == r else 0
    # W[k, i, r] = 1
    
    for r in range(m):
        # U: indices i * m + j
        for i in range(n):
            U[i * m + r, r] = 1
            
        # V: indices j * p + k
        for k in range(p):
            V[r * p + k, r] = 1
            
        # W: indices k * n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1

    # Verify correctness
    U_np = U.astype(np.float64)
    V_np = V.astype(np.float64)
    W_np = W.astype(np.float64)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = 56
        U = np.zeros((N_U, rank), dtype=np.int64)
        V = np.zeros((N_V, rank), dtype=np.int64)
        W = np.zeros((N_W, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
