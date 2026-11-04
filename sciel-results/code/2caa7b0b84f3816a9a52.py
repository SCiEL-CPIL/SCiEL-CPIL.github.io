"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 120
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by eliminating linearly dependent terms
    # Strategy: for each column, check if it can be expressed as a combination of others
    # If so, remove it and adjust the remaining columns
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if the decomposition is valid."""
        n_, m_, p_ = n, m, p
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    val = np.sum(U_mat[i * m_ + j, :] * V_mat[j * p_ + k, :] * W_mat[k * n_ + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Greedy elimination: try to remove each term and see if we can compensate
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Try to find lower rank by checking if columns are linearly dependent
    # in a structured way
    
    # Alternative: use randomization to find a potentially lower rank decomposition
    # For now, try simple elimination
    
    removed = 0
    for attempt in range(current_rank):
        if time.time() > deadline:
            break
        
        # Try to remove the last column
        if current_rank <= 1:
            break
            
        # Check if the last column is linearly dependent on others
        # Build the "effective" contribution
        last_col = current_rank - 1
        
        # Try removing it: check if the tensor can still be represented
        # by the remaining columns (this is hard, so use a heuristic)
        
        # Simple heuristic: if the column has many zeros, it might be removable
        # with adjustment (but this is complex)
        
        # For now, just keep the schoolbook as it's guaranteed correct
        break
    
    # Return as lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Verify the decomposition
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
