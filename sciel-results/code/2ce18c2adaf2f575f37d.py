import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    Uses NumPy vectorized operations to generate the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # We will create the matrices as dense NumPy arrays of zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate the indices for each rank-1 component
    # col is the index in the rank dimension
    # i is 0..n-1
    # j is 0..m-1
    # k is 0..p-1
    
    # We can vectorize this by creating the full set of (i, j, k) tuples
    # There are n*m*p of them.
    
    # Flatten the 3D grid of (i, j, k) into 1D
    # Order: i outer, j middle, k inner (to match the loop in the original code)
    # col = i*m*p + j*p + k
    
    cols = np.arange(rank)
    
    # Reverse map col to i, j, k
    # k = col % p
    # temp = col // p
    # j = temp % m
    # i = temp // m
    
    k_idx = cols % p
    temp = cols // p
    j_idx = temp % m
    i_idx = temp // m
    
    # Calculate flat indices for U, V, W
    # U[(i, j)] -> i*m + j
    u_idx = i_idx * m + j_idx
    
    # V[(j, k)] -> j*p + k
    v_idx = j_idx * p + k_idx
    
    # W[(k, i)] -> k*n + i
    w_idx = k_idx * n + i_idx
    
    # Set the values
    # U[u_idx, col] = 1
    # V[v_idx, col] = 1
    # W[w_idx, col] = 1
    
    U[u_idx, cols] = 1
    V[v_idx, cols] = 1
    W[w_idx, cols] = 1
    
    # Convert to lists to match the expected output format if necessary, 
    # but the task says "factor matrices... with integer/half-integer entries". 
    # The original returned lists of lists. 
    # The evaluator likely accepts NumPy arrays or converts them. 
    # To be safe and consistent with "Python standard library... available", 
    # returning lists is safer if the evaluator doesn't handle arrays. 
    # However, the original code returned lists. 
    # Let's return lists to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
