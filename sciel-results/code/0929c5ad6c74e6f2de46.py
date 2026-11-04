import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized construction of the schoolbook decomposition
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    # Column index c corresponds to the tuple (i, j, k)
    
    # Generate all (i, j, k) tuples
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    col_indices = np.arange(rank)
    
    # Assign 1s to the correct positions
    U[u_indices, col_indices] = 1.0
    V[v_indices, col_indices] = 1.0
    W[w_indices, col_indices] = 1.0
    
    # Convert to list of lists as required by the interface (integer/half-integer entries)
    # The entries are 0 or 1, which are integers.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
