import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as zeros
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    U_np = np.zeros((n * m, rank), dtype=int)
    V_np = np.zeros((m * p, rank), dtype=int)
    W_np = np.zeros((p * n, rank), dtype=int)
    
    # Generate all (i, j, k) indices
    # i: 0..n-1, j: 0..m-1, k: 0..p-1
    # We map each (i, j, k) to a unique rank index r
    # r = i * m * p + j * p + k
    
    # Create index arrays
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten the indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    r_flat = np.arange(rank)
    
    # Compute row indices for U, V, W
    # U index: i * m + j
    u_rows = i_flat * m + j_flat
    # V index: j * p + k
    v_rows = j_flat * p + k_flat
    # W index: k * n + i
    w_rows = k_flat * n + i_flat
    
    # Assign 1s at the correct positions
    U_np[u_rows, r_flat] = 1
    V_np[v_rows, r_flat] = 1
    W_np[w_rows, r_flat] = 1
    
    # Convert to lists to match the interface style of the seed if required, 
    # or return numpy arrays if acceptable. 
    # The seed returns lists of lists. I will return lists to be safe.
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
