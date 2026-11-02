import numpy as np

def construct():
    n, m, p = 2, 3, 9
    
    # The task is to decompose the matrix multiplication tensor for (n x m) * (m x p).
    # We view the output as p independent columns, each of size n x 1.
    # Each column k is the result of multiplying the (n x m) matrix by the k-th column (m x 1) of the second matrix.
    # This corresponds to the tensor T = direct_sum_{k=1}^p T_{n,m,1}.
    
    # The rank of the (n, m, 1) tensor is n * m (schoolbook is optimal for 2x3x1).
    # So for each of the p columns, we need rank n * m = 6.
    # Total rank = p * n * m = 9 * 2 * 3 = 54.
    
    rank = n * m * p
    
    # Initialize matrices as NumPy arrays of zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We will fill the matrices by iterating over the p independent blocks (columns of the output).
    # For each block k (0 <= k < p):
    #   The block corresponds to the multiplication A (n x m) * v_k (m x 1) = w_k (n x 1).
    #   The tensor for this block has rank n * m.
    #   We assign the next n * m columns of U, V, W to this block.
    
    for k in range(p):
        # The slice of columns for this block
        start_col = k * (n * m)
        end_col = start_col + (n * m)
        block_slice = slice(start_col, end_col)
        
        # Within this block, we have n * m terms corresponding to the standard dot product terms.
        # The term for A[i, j] * v_k[j] contributes to W[k, i].
        # In the decomposition sum_r U[a, r] V[b, r] = W[c, r], 
        # for the k-th block, we assign r from start_col to end_col-1.
        
        # We map the local index in the block (0 to n*m-1) to global index.
        # Local index l corresponds to a specific (i, j) pair.
        # Let's iterate i in 0..n-1 and j in 0..m-1.
        # The global column index is start_col + i * m + j.
        
        for i in range(n):
            for j in range(m):
                r = start_col + i * m + j
                
                # U index: row corresponding to A[i, j] is i * m + j
                U[i * m + j, r] = 1
                
                # V index: row corresponding to B[j, k] is j * p + k
                V[j * p + k, r] = 1
                
                # W index: row corresponding to C[k, i] (trace convention: W[k*n+i, r]) is k * n + i
                W[k * n + i, r] = 1
                
    # Convert back to list of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
