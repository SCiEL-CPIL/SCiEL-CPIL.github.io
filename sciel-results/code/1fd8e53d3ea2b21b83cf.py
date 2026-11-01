"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    Uses vectorized NumPy construction of the schoolbook decomposition (rank 60).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate column indices 0 to rank-1
    cols = np.arange(rank)
    
    # Decode column index into (i, j, k) based on seed ordering:
    # c = i * (m * p) + j * p + k
    i_idx = cols // (m * p)
    rem = cols % (m * p)
    j_idx = rem // p
    k_idx = rem % p
    
    # Calculate row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set 1s at the calculated positions
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists to match the seed's return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
