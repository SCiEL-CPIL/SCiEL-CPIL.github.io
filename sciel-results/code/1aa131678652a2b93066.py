import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for 2x5x5 matrix multiplication.
    Returns U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition is exact and uses rank n*m*p = 50.
    """
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Generate indices for the decomposition
    # Each term in the decomposition corresponds to a triplet (i, j, k)
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1

    # Create arrays of indices
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)

    # Construct the factor matrices directly using these indices
    # U is (n*m, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Map indices to row positions
    u_rows = i_vals * m + j_vals
    v_rows = j_vals * p + k_vals
    w_rows = k_vals * n + i_vals

    # Set the ones
    cols = np.arange(rank)
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1

    if time.time() > deadline:
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
