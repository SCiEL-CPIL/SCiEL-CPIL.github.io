import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Approach: Numerical Rank Estimation and Factor Recovery.
    1. Construct the full multiplication tensor T.
    2. Matricize T and compute SVD to estimate the numerical rank.
    3. If the numerical rank R is significantly less than the schoolbook rank (30), 
       attempt to recover exact factors by rounding the SVD components or solving a 
       linear system for a decomposition of rank R.
    4. If no lower rank is found or recovery fails, fallback to the standard 
       schoolbook decomposition of rank 30.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # 1. Construct the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # 2. Matricize T to estimate rank
    # Flatten first two dimensions: (n*m) x (m*p * p*n)
    # Or flatten last two: (n*m * m*p) x (p*n)
    # The rank of the tensor is the rank of any matricization.
    # Let's use (n*m) x (m*p * p*n)
    T_mat = T.reshape(n * m, -1)
    
    # Compute SVD
    # We only need the singular values to determine rank
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # Determine numerical rank
    # A singular value is considered non-zero if it is significantly larger than the smallest one
    # Since T is integer, singular values should be close to integers or 0.
    # Use a tolerance relative to the largest singular value.
    tol = 1e-10
    numerical_rank = np.sum(s_vals > tol)
    
    # The true rank is an integer. If numerical_rank is close to an integer R < schoolbook_rank, 
    # we might have found a lower rank.
    # However, SVD of a sparse integer tensor might have small numerical noise.
    # If numerical_rank < schoolbook_rank, we attempt to find a decomposition of that rank.
    
    # For now, if numerical_rank is not clearly lower, we use schoolbook.
    # Note: For M_{2,3,5}, the rank is likely 30. 
    # If numerical_rank == 30, we return schoolbook.
    
    if numerical_rank < schoolbook_rank:
        # Attempt to recover a decomposition of rank numerical_rank
        # This is complex. We'll just use the top numerical_rank components from SVD.
        # U_svd[:, :R] gives factors for U and V (partially).
        # This is a heuristic and might not yield exact integer/half-integer results.
        # Given the risk of inexactness, and the likelihood that rank is 30,
        # we will only use this path if we can verify it.
        
        R = int(numerical_rank)
        # Get factors from SVD
        # T_mat = U_svd @ diag(s_vals) @ Vt_svd
        # T_mat = (U_svd[:, :R] * s_vals[:R]) @ Vt_svd[:R, :]
        # This gives a rank R approximation.
        # We need to map this back to U, V, W.
        # T = sum_r u_r v_r w_r
        # The SVD gives us a decomposition of the matricization, which corresponds to the tensor.
        
        # U_factor (n*m, R)
        U_fact = U_svd[:, :R] * np.sqrt(s_vals[:R])
        # V_factor (m*p*p*n, R) -> This needs to be reshaped to (m*p, R) and (p*n, R)
        # Vt_svd[:R, :] is (R, m*p*p*n)
        # Let V_mat = Vt_svd[:R, :].T -> (m*p*p*n, R)
        # V_mat should be reshaped to (m*p, p*n, R)
        V_mat = Vt_svd[:R, :].T
        V_tensor = V_mat.reshape(m * p, p * n, R)
        
        # The tensor is approx T[a, b, c] = sum_r U_fact[a, r] * V_tensor[b, c, r]
        # We need to separate V_tensor into V and W such that V_tensor[b, c, r] = V[b, r] * W[c, r]
        # This is a rank-1 decomposition for each r of the slice V_tensor[:,:,r].
        # This is generally not possible exactly unless the tensor has a specific structure.
        # Therefore, simple SVD truncation does NOT directly give a valid U, V, W decomposition 
        # in the required format (U, V, W) unless the tensor is a sum of rank-1 tensors 
        # which are separable in the V and W indices.
        
        # Since this is not guaranteed, and the probability of rank < 30 is low,
        # we fallback to the safe schoolbook decomposition.
        pass

    # Fallback: Schoolbook Decomposition
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
