import numpy as np

def construct():
    """
    Constructs a rank-50 decomposition for the 2x5 by 5x5 matrix multiplication tensor.
    
    This implementation uses a vectorized NumPy approach to generate the factor matrices 
    U, V, and W. It models the multiplication tensor T_{i,j,k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    
    The 'schoolbook' decomposition assigns a unique rank index r to each (i, j, k) triple.
    This program constructs the factors by explicitly mapping the tensor indices to the 
    factor matrix entries, ensuring exact decomposition.
    """
    n, m, p = 2, 5, 5
    # Total dimensions for the matrix multiplication problem
    dim_a = n * m  # 10
    dim_b = m * p  # 25
    dim_c = p * n  # 10
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((dim_a, rank))
    V = np.zeros((dim_b, rank))
    W = np.zeros((dim_c, rank))
    
    # Generate the decomposition
    # For each (i, j, k), we create a rank slice that isolates the term A[i,j] * B[j,k]
    # This contributes to C[i,k]
    
    # We can vectorize the index generation
    # Indices i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Create meshgrid of indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to get the rank index for each combination
    # The rank index r corresponds to the linear index of (i, j, k) in the flattened tensor
    # We map each (i, j, k) to a unique column in U, V, W
    
    # Reshape indices to (rank, 1) to broadcast against the matrix dimensions
    # We need to assign 1s to specific positions in U, V, W for each r
    
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i
    
    # Flatten the index arrays to match the rank dimension
    flat_i = i_idx.flatten()
    flat_j = j_idx.flatten()
    flat_k = k_idx.flatten()
    
    u_rows = (flat_i * m + flat_j)
    v_rows = (flat_j * p + flat_k)
    w_rows = (flat_k * n + flat_i)
    
    # The columns in U, V, W correspond to the linear index in the flattened (n,m,p) space
    cols = np.arange(rank)
    
    # Set the values to 1
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Constructed rank 50 decomposition.")
