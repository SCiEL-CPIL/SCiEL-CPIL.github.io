def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    rank = m * p  # 12
    
    # U: shape (n*m, rank) = (6, 12)
    # U[i*m+j, r] = 1 if r // p == j else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if r // p == j:
                    U[i * m + j][r] = 1
    
    # V: shape (m*p, rank) = (12, 12)
    # V[j*p+k, r] = 1 if r == j*p + k else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j * p + k] = 1
    
    # W: shape (p*n, rank) = (8, 12)
    # W[k*n+i, r] = 1 if r % p == k else 0
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                if r % p == k:
                    W[k * n + i][r] = 1
    
    # Verify
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                assert abs(val - 1.0) < 1e-9, f"Verification failed for ({i},{j},{k}): {val}"
    
    return {"U": U, "V": V, "W": W}
