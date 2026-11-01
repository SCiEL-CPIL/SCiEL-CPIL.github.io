import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition is exact."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    if rank != V_arr.shape[1] or rank != W_arr.shape[1]:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    
    # Check invalid entries
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m + j, j*p + k, k*n + i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in valid:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(s) > 1e-10:
                    return False
    
    return True

def construct():
    n, m, p = 3, 3, 3
    
    # Schoolbook decomposition (Rank 27)
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
