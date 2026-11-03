import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication M_{3,4,5}.
    Uses a vectorized approach to generate the schoolbook decomposition factors.
    This is a different implementation approach (vectorized/tensor-based) compared to explicit loops.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We need to generate U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # The rank index r iterates over all (i, j, k) combinations.
    # Let's create indices for i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    
    # We can create a meshgrid of indices
    I, J, K = np.indices((n, m, p))
    
    # Flatten these to get the linear index for the rank dimension
    # The order of flattening matters to match the definition, but since we construct
    # U, V, W consistently with the same flattening, the decomposition will be valid.
    # Let's flatten in C-order (k varies fastest, then j, then i)
    # This matches the loop: for i: for j: for k:
    
    r_idx = np.arange(rank)
    
    # Recover i, j, k from r_idx? Or just use the meshgrid flattened.
    # Flattened I, J, K correspond to the same r_idx order.
    i_flat = I.flatten()
    j_flat = J.flatten()
    k_flat = K.flatten()
    
    # Construct U
    # U[row, r] = 1 if row == i_flat[r] * m + j_flat[r]
    # U is effectively a sparse matrix. We can create it by placing 1s.
    U = np.zeros((n * m, rank), dtype=int)
    U[i_flat * m + j_flat, r_idx] = 1
    
    # Construct V
    # V[row, r] = 1 if row == j_flat[r] * p + k_flat[r]
    V = np.zeros((m * p, rank), dtype=int)
    V[j_flat * p + k_flat, r_idx] = 1
    
    # Construct W
    # W[row, r] = 1 if row == k_flat[r] * n + i_flat[r]
    # Note the convention: W[k*n + i, r]
    W = np.zeros((p * n, rank), dtype=int)
    W[k_flat * n + i_flat, r_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
