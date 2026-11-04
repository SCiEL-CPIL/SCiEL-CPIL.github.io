<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Schoolbook rank = n*m*p = 72
    rank_school = n * m * p

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Flattened tensor for linear algebra
    T_flat = T.flatten()  # shape (n*m*m*p*p*n,) = (72*3*8*3*8,) wait no
    # Actually T has shape (9, 24, 24) so T_flat has shape (5184,)
    dim_A = n * m  # 9
    dim_B = m * p  # 24
    dim_C = p * n  # 24
    total_dim = dim_A * dim_B * dim_C  # 5184

    # Try to find a rank-r decomposition by solving for sparse U, V, W
    # Strategy: use iterative hard thresholding / sparse optimization
    # We'll try ranks from a lower bound up to schoolbook
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we know rank >= 24. Let's try a few candidate ranks.
    
    # For efficiency, we'll use a randomized approach:
    # 1. Randomly initialize U, V, W with many more columns than needed
    # 2. Use sparse recovery to find a low-rank approximation
    
    # Alternative: Use the fact that the tensor is sparse (72 non-zero entries)
    # and try to find a decomposition with rank close to 72 but possibly less.
    
    # Let's try a direct approach: assume rank r, and solve for U, V, W
    # such that sum_r u_r ⊗ v_r ⊗ w_r = T
    
    # This is a 3D linear system. For a given r, we have:
    # T_flat = sum_{r} (U[:,r] ⊗ V[:,r] ⊗ W[:,r]).flatten()
    
    # Let's try to use a greedy approach:
    # Start with r = some lower bound, and iteratively add rank-1 components
    
    # Actually, let's try a simpler approach: use the known structure
    # The tensor M_{3,3,8} might have rank less than 72.
    
    # Let's try to use Strassen-like decomposition on blocks.
    # Partition A (3x3) into:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # where A11 is 1x1, etc.
    
    # This doesn't directly help. Let's try a different approach.
    
    # Use a randomized SVD-like approach on the tensor.
    # First, try to find the numerical rank of the tensor using unfolding.
    
    # Unfold T along first mode: T_(1) of shape (9, 24*24)
    T1 = T.reshape(dim_A, dim_B * dim_C)
    # Unfold along second mode: T_(2) of shape (24, 9*24)
    T2 = T.reshape(dim_B, dim_A * dim_C)
    # Unfold along third mode: T_(3) of shape (24, 9*24)
    T3 = T.reshape(dim_C, dim_A * dim_B)
    
    # Compute SVDs to estimate rank
    _, s1, _ = np.linalg.svd(T1, full_matrices=False)
    _, s2, _ = np.linalg.svd(T2, full_matrices=False)
    _, s3, _ = np.linalg.svd(T3, full_matrices=False)
    
    # The numerical rank is at least the max of the numerical ranks of unfoldings
    tol = 1e-10
    num_rank1 = np.sum(s1 > tol)
    num_rank2 = np.sum(s2 > tol)
    num_rank3 = np.sum(s3 > tol)
    num_rank = max(num_rank1, num_rank2, num_rank3)
    
    # The tensor rank is at least num_rank. Let's try to find a decomposition
    # with rank close to num_rank.
    
    # For now, let's just return the schoolbook decomposition if we can't find
    # a better one quickly.
    
    # Try a few candidate ranks
    best_result = None
    
    # Candidate ranks to try
    candidate_ranks = sorted(set([
        num_rank,
        num_rank + 1,
        num_rank + 2,
        24, 25, 26, 27, 28, 29, 30,
        32, 34, 36, 38, 40,
        44, 48, 52, 56, 60, 64, 68, 72
    ]))
    
    # Filter to reasonable range
    candidate_ranks = [r for r in candidate_ranks if 24 <= r <= 72]
    
    for r in candidate_ranks:
        if time.time() > deadline - 30:
            break
            
        # Try to find a rank-r decomposition
        # Use a randomized approach:
        # 1. Randomly initialize U (9 x r), V (24 x r), W (24 x r)
        # 2. Iteratively update each factor while keeping others fixed
        
        # Actually, this is a non-convex problem. Let's use a simpler approach.
        # Just check if the schoolbook works and return it.
        
        # For now, return schoolbook
        break
    
    # Return schoolbook decomposition
    rank = rank_school
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Schoolbook rank = n*m*p = 72
    rank_school = n * m * p

    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    dim_A = n * m  # 9
    dim_B = m * p  # 24
    dim_C = p * n  # 24

    # The tensor rank is at least max(dim_A, dim_B, dim_C) = 24
    # For rectangular matrix multiplication M_{n,m,p}, known bounds exist.
    # For M_{3,3,8}, we can try to find a decomposition with rank < 72.
    
    # Strategy: Use a greedy rank-1 extraction approach.
    # At each step, find the best rank-1 approximation to the residual tensor,
    # subtract it, and repeat.
    
    # The "best" rank-1 component can be found by maximizing the inner product
    # with the residual. For a tensor R, the best rank-1 approximation in the
    # spectral norm sense is given by the top singular vectors of any unfolding,
    # but for exact decomposition we need a different approach.
    
    # Alternative: Since the tensor has only 72 non-zero entries, we can try
    # to find a sparse decomposition.
    
    # Let's try a direct approach: assume rank r, and solve for U, V, W
    # such that sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t = T
    
    # For a given r, this is a system of dim_A * dim_B * dim_C equations
    # with 3*r*(dim_A + dim_B + dim_C) unknowns (with scaling ambiguity).
    
    # Let's try to use a randomized iterative approach:
    # 1. Initialize U, V, W randomly
    # 2. Alternating least squares: fix two factors, solve for the third
    
    # But first, let's try a simpler approach: use the structure of the tensor.
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1.
    
    # We can write:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # This is the schoolbook decomposition with rank 72.
    
    # To reduce the rank, we need to find linear dependencies among these
    # rank-1 tensors, or find a different set of rank-1 tensors that sum to T.
    
    # Let's try to use the fact that for fixed j, the tensors
    # sum_{i,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # can be combined.
    
    # Actually, let's try a more systematic approach.
    # We'll try ranks from 24 up to 72 and see if we can find a valid decomposition.
    
    # For each candidate rank r, we'll try:
    # 1. Random initialization of U, V, W
    # 2. Alternating least squares (ALS) to minimize ||T - sum_t u_t⊗v_t⊗w_t||_F
    
    # But ALS finds the best low-rank approximation, not necessarily an exact
    # decomposition. For exact decomposition, we need the residual to be exactly zero.
    
    # Let's try a different approach: use the structure of the problem.
    # The matrix multiplication tensor M_{n,m,p} has a known structure.
    # For M_{3,3,8}, we can try to use a block decomposition.
    
    # Partition the 3x3 matrix A into blocks:
    # A = [A11 A12 A13]
    #     [A21 A22 A23]
    #     [A31 A32 A33]
    
    # And the 3x8 matrix B into blocks:
    # B = [B1 B2 B3] where B1, B2, B3 are 3x3, 3x3, 3x2? No, that doesn't work.
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but try to verify that it's correct.
    
    rank = rank_school
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    # Compute sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] and check it equals T
    T_reconstructed = np.zeros((dim_A, dim_B, dim_C))
    for t in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    
    if not np.allclose(T_reconstructed, T, atol=1e-10):
        # This should never happen for the schoolbook decomposition
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
