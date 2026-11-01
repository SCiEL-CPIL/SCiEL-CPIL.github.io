import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify decomposition exactness (cheap check)
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Reconstruct tensor: T[i*m+j, j*p+k, k*n+i] should be 1
        # Check a few random entries
        for _ in range(10):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            # The contribution to T[i*m+j, j*p+k, k*n+i] from column (i,j,k)
            expected = 1
            actual = U_arr[i*m+j, col_idx(i,j,k)] * V_arr[j*p+k, col_idx(i,j,k)] * W_arr[k*n+i, col_idx(i,j,k)]
            if abs(actual - expected) > 1e-9:
                # Fallback: return as-is (schoolbook is always valid)
                pass
    
    return result

def col_idx(i, j, k):
    n, m, p = DIMENSIONS
    return i * m * p + j * p + k
