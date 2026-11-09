"""Matrix multiplication 3x4x6 tensor decomposition via numpy construction."""
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for the rank-1 factorization
    # We construct U, V, W as (n*m, total), (m*p, total), (p*n, total)
    
    # Create index arrays for each mode
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)

    # Flatten indices into row/column indices for U, V, W
    u_rows = i_idx * m + j_idx       # shape (total,)
    v_rows = j_idx * p + k_idx       # shape (total,)
    w_rows = k_idx * n + i_idx       # shape (total,)

    # Build sparse-like matrices by scattering 1s
    U = np.zeros((n * m, total), dtype=float)
    V = np.zeros((m * p, total), dtype=float)
    W = np.zeros((p * n, total), dtype=float)

    U[u_rows, np.arange(total)] = 1.0
    V[v_rows, np.arange(total)] = 1.0
    W[w_rows, np.arange(total)] = 1.0

    # Verify correctness: sum_r U[:,r] * V[:,r] should give the multiplication tensor
    # Reconstruct C from A and B symbolically check: 
    # For the tensor, we need that for all (i,j) in U-space, (j',k') in V-space, (k'',i'') in W-space:
    # The product should yield 1 iff j == j' and i == i'' and k == k''
    # This is guaranteed by construction since each column r has exactly one 1 in each of U, V, W
    # at positions corresponding to a valid (i,j,k) triple.

    # Convert to lists for the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
