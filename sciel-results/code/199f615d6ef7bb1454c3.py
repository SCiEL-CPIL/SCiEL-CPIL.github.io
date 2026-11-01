"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    
    The decomposition uses the schoolbook algorithm as a guaranteed valid baseline
    with rank = n*m*p = 36. We attempt to find a lower rank decomposition
    within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    target_rank = n * m * p  # 36 for schoolbook
    
    # Build the matrix multiplication tensor T[i, j, k] where
    # T is the trilinear form: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # Actually, the tensor is T[i,j,k] = delta[i,j] * delta[j,k] in the right way.
    
    # The matrix multiplication tensor for (n,m,p) has entries:
    # T[i,j,k] = 1 if (i,j) is a valid (row, col) of A and (j,k) is valid (col, row) of B
    # More precisely, T_{i,j,k} = 1 if j is the shared dimension index
    # In our case: i in [0,n), j in [m, p), k in [p, n)
    # T[i,j,k] = 1 if the multiplication at position (i,j,k) is part of the product
    
    # The standard decomposition has rank n*m*p with:
    # U[i*m+j, c] = 1 if c corresponds to (i,j)
    # V[j*p+k, c] = 1 if c corresponds to (j,k)  
    # W[k*n+i, c] = 1 if c corresponds to (i,k)
    
    # Let's first provide the schoolbook solution, then try to optimize
    rank = target_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_result = {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
    
    # Try to find a lower rank decomposition using randomized methods
    # Check if we have enough time
    if time.time() > deadline - 10:
        return best_result
    
    # The tensor T can be viewed as a matrix unfolding
    # Let's try to use rank factorization approaches
    
    # Flatten the tensor T[i,j,k] into a matrix
    # T as a 3D array of shape (n, m, p) where T[i,j,k] = 1 if valid
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # All combinations are valid for matrix mult
    
    # Try random rank factorizations to see if lower rank works
    # For each candidate rank r, try to find U, V, W such that
    # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k]
    
    best_rank = target_rank
    
    # Try ranks from lower to higher
    for r in range(30, target_rank + 1):  # Try ranks 30-36
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        max_trials = 50 if r >= 32 else 20
        
        for trial in range(max_trials):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            np.random.seed(42 + trial)
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # Verify the decomposition
            # Reconstruct tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            reconstructed = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U_rand[i * m + j, c] * V_rand[j * p + k, c] * W_rand[k * n + i, c]
                        reconstructed[i, j, k] = val
            
            # Check if reconstruction matches T
            if np.allclose(reconstructed, T, atol=1e-6):
                found = True
                break
        
        if found:
            best_rank = r
            # Convert to lists
            best_result = {
                'U': U_rand.tolist(),
                'V': V_rand.tolist(),
                'W': W_rand.tolist()
            }
            break
    
    return best_result
# EVOLVE-BLOCK-END
