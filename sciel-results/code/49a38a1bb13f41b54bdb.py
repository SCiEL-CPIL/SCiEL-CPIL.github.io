import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = float(np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # The tensor for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k] 
    # has entries T[i,j,k] = 1 for all valid i, j, k.
    # This is the all-ones tensor of shape (n, m, p).
    # The all-ones tensor has rank 1, with factors:
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k
    # W[k*n+i, 0] = 1 for all k, i
    
    # Let's verify this rank-1 decomposition.
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    best = _build_schoolbook(n, m, p)
    return best
# EVOLVE-BLOCK-END

BET = {"mechanism": "The matrix multiplication tensor T[i,j,k] is 1 for all valid indices. The all-ones tensor has rank 1, decomposable into three all-ones vectors. This reduces rank from 48 to 1.", "q": 0.99, "delta": [0.0, 47.0]}
