"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create numpy arrays for efficiency in construction
    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices based on the tensor structure
    # T = sum_{i,j,k} e_{i,j}^A \otimes e_{j,k}^B \otimes e_{k,i}^C (with W index k*n+i)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U_np[u_idx, col] = 1
                V_np[v_idx, col] = 1
                W_np[w_idx, col] = 1
                col += 1
                
    # Convert back to lists of lists as required by the interface
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
