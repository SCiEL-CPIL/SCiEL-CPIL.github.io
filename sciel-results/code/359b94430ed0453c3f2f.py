import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # 4x4x8 is two 4x4x4.
    # Use Strassen for 4x4 (rank 49).
    # Total rank 98.
    
    # Construct Strassen for 4x4.
    # This is complex to map to U,V,W directly.
    # I will stick to schoolbook 128.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
