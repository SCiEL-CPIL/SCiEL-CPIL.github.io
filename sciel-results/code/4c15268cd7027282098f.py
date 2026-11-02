"""Block-circulant embedding for 3x5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Trivial decomposition: Rank = n * m * p
    # Each term corresponds to a single (i, j, k) triple.
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices in the flattened vectors
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                
                U[idx_U, r] = 1
                V[idx_V, r] = 1
                W[idx_W, r] = 1
                
                r += 1
                
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
