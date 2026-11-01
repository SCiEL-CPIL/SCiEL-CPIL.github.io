import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m by m x p.
    
    Currently implements the schoolbook algorithm (rank = n*m*p) using NumPy
    for efficient construction. This serves as a baseline that can be improved
    by substituting the decomposition logic with a lower-rank bilinear algorithm.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the matrices according to the schoolbook decomposition:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Tensor term (i, j, k) corresponds to U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as required by the interface (integer/half-integer entries)
    # The problem statement allows integer/half-integer. 1 is an integer.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
