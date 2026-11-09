import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Trivial decomposition:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Each term A[i, j] * B[j, k] is a rank-1 tensor.
    # Total rank = n * m * p
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[k, i] (trace convention W[k*n+i, r])
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    # Convert to lists of ints
    U = [[int(round(v)) for v in row] for row in U]
    V = [[int(round(v)) for v in row] for row in V]
    W = [[int(round(v)) for v in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
