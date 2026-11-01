"""Optimized construction for 2x3x8 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Pre-allocate numpy arrays for efficiency during construction
    # Then convert to lists as per original interface
    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)
    
    # Generate indices for the schoolbook decomposition
    # Each term r corresponds to a unique (i, j, k)
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    for r in range(rank):
        # Decompose r into i, j, k
        # r = i * (m * p) + j * p + k
        k = r % p
        tmp = r // p
        j = tmp % m
        i = tmp // m
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_np[u_idx, r] = 1
        V_np[v_idx, r] = 1
        W_np[w_idx, r] = 1
        
    # Convert to lists of lists to match the expected output format of the seed
    U_list = U_np.tolist()
    V_list = V_np.tolist()
    W_list = W_np.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
