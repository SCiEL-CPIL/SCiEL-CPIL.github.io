import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for easier manipulation
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct the decomposition
    # The tensor T is defined such that T[i, j, k] is the coefficient of the (i, k)-th entry of the product
    # in the term corresponding to the j-th element of the intermediate vector.
    # Specifically, (A B)_{i, k} = sum_j A_{i, j} B_{j, k}
    # The decomposition is sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} ? No.
    # The standard decomposition is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # This is a sum of rank-1 tensors where each term is A_{i, j} * B_{j, k} * 1? No.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k \cdot e_i \otimes e_k ?
    # Let's stick to the definition:
    # We want matrices U, V, W such that
    # sum_c U_{row_U, c} V_{row_V, c} W_{row_W, c} reconstructs the multiplication.
    # The indices for U are (i, j) flattened to i*m+j.
    # The indices for V are (j, k) flattened to j*p+k.
    # The indices for W are (k, i) flattened to k*n+i.
    
    # For each (i, j, k), we have a term that contributes 1 to the (i, k) entry if we multiply
    # U[i, j] * V[j, k] * W[k, i].
    # In the flattened matrices:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # We can vectorize the index generation.
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten the indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate flattened row indices
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # The column index is simply the index in the flattened arrays
    cols = np.arange(rank)
    
    # Assign 1s
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists as required by the interface (though numpy arrays are often accepted,
    # the seed used lists, so we follow suit to be safe)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
