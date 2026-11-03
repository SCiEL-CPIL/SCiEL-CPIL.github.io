import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    # T[i, j, k] = 1 if the indices correspond to a valid term
    # Index mapping:
    # U index: i * m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j * p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k * n + i  (k in 0..p-1, i in 0..n-1)
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
                
    # Flatten T for linear algebra
    T_flat = T.flatten()
    
    # Function to check if a decomposition is exact
    def check_decomposition(U, V, W):
        # U: (dim_u, r)
        # V: (dim_v, r)
        # W: (dim_w, r)
        # Reconstruct T
        # T_recon[i, j, k] = sum_l U[i, l] * V[j, l] * W[k, l]
        # We can compute this efficiently
        # For each l, add U[:, l] * V[:, l] * W[:, l]
        # This is a sum of rank-1 tensors.
        
        # To speed up, we can use einsum or manual loops
        # Given small dims, manual loop over r is fine.
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for l in range(U.shape[1]):
            u_col = U[:, l]
            v_col = V[:, l]
            w_col = W[:, l]
            # Outer product of 3 vectors
            # T_recon += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
            # Or:
            T_recon += np.outer(np.outer(u_col, v_col), w_col)
            
        if np.array_equal(T_recon, T):
            return True
        return False

    # Search for lower rank
    # Start from 60 down to 40
    # Use a time limit
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = 60
    
    # Initialize with schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_v, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_w, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a better rank
    # We will try random searches for ranks 55, 50, 45
    for target_rank in [55, 50, 45]:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V with small random integers
            U = np.random.randint(-2, 3, size=(dim_u, target_rank))
            V = np.random.randint(-2, 3, size=(dim_v, target_rank))
            
            # Solve for W
            # T_flat = sum_l (U[:, l] * V[:, l]) * W[:, l]
            # Let A_l = U[:, l] * V[:, l] (element-wise product, size dim_u * dim_v)
            # T_flat is size dim_u * dim_v * dim_w
            # This is not a simple linear system in W because W is multiplied element-wise in the 3rd dimension.
            # T[i, j, k] = sum_l U[i, l] V[j, l] W[k, l]
            # For a fixed i, j, the equation is:
            # T[i, j, :] = sum_l (U[i, l] V[j, l]) W[:, l]
            # This is a linear system in W for each pair (i, j).
            # We have dim_u * dim_v such systems.
            # We can combine them into one large system.
            
            # Let's build the matrix M such that M * W_flat = T_flat
            # M will have size (dim_u * dim_v * dim_w) x (dim_w * target_rank)
            # This is too large to build explicitly if target_rank is large?
            # dim_u=12, dim_v=20, dim_w=15. Total entries 3600.
            # W has 15 * target_rank entries.
            # If target_rank=50, W has 750 entries.
            # M is 3600 x 750. This is manageable.
            
            M = np.zeros((dim_u * dim_v * dim_w, dim_w * target_rank), dtype=np.int64)
            
            # Fill M
            # For each i, j, k, and each l:
            # T[i, j, k] = sum_l U[i, l] V[j, l] W[k, l]
            # The coefficient of W[k, l] in the equation for T[i, j, k] is U[i, l] V[j, l].
            
            for i in range(dim_u):
                for j in range(dim_v):
                    for k in range(dim_w):
                        row_idx = (i * dim_v + j) * dim_w + k
                        for l in range(target_rank):
                            col_idx = k * target_rank + l
                            M[row_idx, col_idx] = U[i, l] * V[j, l]
                            
            # Now solve M * W_flat = T_flat
            # We use least squares
            try:
                W_flat, residuals, rank_M, s = np.linalg.lstsq(M, T_flat, rcond=None)
            except:
                continue
                
            W = W_flat.reshape(dim_w, target_rank)
            
            # Check if the solution is exact
            if check_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break # Found a solution for this rank, try lower?
                
            # If not exact, we can try to refine or just move to next attempt
            # The solution from lstsq minimizes error, but might not be integer/exact.
            # We can round W and check again?
            W_rounded = np.round(W).astype(np.int64)
            if check_decomposition(U, V, W_rounded):
                best_U, best_V, best_W = U, V, W_rounded
                best_rank = target_rank
                break
                
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
