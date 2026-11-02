import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # Split p=7 into p1=3 and p2=4
    p1 = 3
    p2 = 4
    
    # Rank for 3x3 part (trivial)
    rank1 = n * m * p1
    # Rank for 3x4 part (trivial)
    rank2 = n * m * p2
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: 3x3 multiplication (k in 0..2)
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Part 2: 3x4 multiplication (k in 3..6)
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Construct target tensor indices directly for comparison
    # T_target[a, b, c] is 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # This is exactly the structure of the trivial decomposition.
    # We can verify by checking if the computed tensor matches the identity-like structure.
    
    T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check specific entries that must be 1
    # and others that must be 0
    # For efficiency, we can check a subset or rely on the construction.
    # Given the integer nature and small size, a full check is fast.
    
    # Create target
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    if not np.allclose(T_comp, T_target, atol=1e-10):
        raise ValueError("Verification failed")

    return {"U": U, "V": V, "W": W}
