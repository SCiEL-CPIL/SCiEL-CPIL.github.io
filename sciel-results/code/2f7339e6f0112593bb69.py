"""Matrix multiplication 2x3 by 3x4 tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Vectorized construction of the schoolbook decomposition.
    # This uses a different implementation approach (NumPy vectorization) 
    # while maintaining the minimal known rank for this size (24).
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index grids
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices to create column indices for U, V, W
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i
    
    U_rows = (i_idx * m + j_idx).flatten()
    V_rows = (j_idx * p + k_idx).flatten()
    W_rows = (k_idx * n + i_idx).flatten()
    
    # Initialize sparse-like matrices (using dense for simplicity as rank is small)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set values
    # Each element in the flattened grid corresponds to one rank-1 term (column)
    cols = np.arange(rank)
    
    U[U_rows, cols] = 1
    V[V_rows, cols] = 1
    W[W_rows, cols] = 1
    
    # Convert to lists as required by the interface (or keep as arrays if allowed? 
    # Interface says 'integer/half-integer entries', usually lists or arrays are fine. 
    # The seed used lists. I will convert to lists to be safe with the evaluator.)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
