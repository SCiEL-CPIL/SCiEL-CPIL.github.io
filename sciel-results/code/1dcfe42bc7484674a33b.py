"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses NumPy for efficient generation of the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) triples
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Calculate linear indices for U, V, W
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i
    u_indices = (i_vals * m + j_vals).flatten()
    v_indices = (j_vals * p + k_vals).flatten()
    w_indices = (k_vals * n + i_vals).flatten()
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate matrices
    cols = np.arange(rank)
    U[u_indices, cols] = 1
    V[v_indices, cols] = 1
    W[w_indices, cols] = 1
    
    # Convert to lists to match the interface expectation of the seed program
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
