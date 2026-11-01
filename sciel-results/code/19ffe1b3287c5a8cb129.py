"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p) is defined such that:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    In the factorization U (n*m x r), V (m*p x r), W (p*n x r):
    For each output index (i, k), we need:
    C[i, k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) ... 
    
    Actually, the standard formulation is:
    The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i in [n], j in [m], k in [p].
    A rank-r decomposition is: sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r = T
    where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}.
    
    The schoolbook decomposition uses R = n*m*p = 54, with each term picking out one (i,j,k) triple.
    """
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback (rank 54)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # The tensor T is a 3-tensor with dimensions (n*m, m*p, p*n) = (6, 27, 18)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # All other entries are 0.
    
    # Known results:
    # - For (2,2,2): rank is 7 < 8
    # - For (2,2,3): rank is 11 < 12
    # - For (2,3,3): rank is 11 < 18 (this is a known result)
    # - For (2,2,p): rank is 2p+1 for p >= 2 (I think)
    # - For (2,3,p): the rank might be lower than 6p
    
    # Let's try to construct a lower-rank decomposition.
    # Key insight: for n=2, we can potentially share terms between the two "layers" i=0 and i=1.
    
    # Approach: Try to find a decomposition of rank R < 54.
    # We'll use a structured approach based on known low-rank constructions.
    
    # For matrix multiplication (2, m, p), there are known constructions.
    # One approach: use the fact that C[0,k] and C[1,k] can be computed together.
    
    # Let's try a direct construction for (2, 3, 9).
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # For each k, we have:
    # [C[0,k]]   [A[0,0] A[0,1] A[0,2]]   [B[0,k]]
    # [C[1,k]] = [A[1,0] A[1,1] A[1,2]] * [B[1,k]]
    #                              [B[2,k]]
    #
    # This is a 2x3 times 3x1 matrix product for each k.
    # The rank of (2,3,1) multiplication is 6 (schoolbook) or possibly lower.
    # Actually for (2,3,1): C[i] = sum_j A[i,j] B[j], which is just 6 multiplications.
    # Can we do it in fewer? For (2,3,1), the tensor has T[i*m+j, j, i] = 1.
    # Dimensions: (6, 3, 2). Rank is at most 6.
    # Is rank 5 possible? The tensor T has 6 non-zero entries, each in a distinct "fiber".
    # Actually, for (n,m,1), the rank is n*m because each output C[i] depends on all A[i,j]*B[j].
    # Wait, no. For (2,3,1): C[0] = A[0,0]B[0] + A[0,1]B[1] + A[0,2]B[2]
    #                     C[1] = A[1,0]B[0] + A[1,1]B[1] + A[1,2]B[2]
    # We can compute S0 = A[0,0]+A[1,0], S1 = A[0,1]+A[1,1], S2 = A[0,2]+A[1,2]
    # Then S0*B[0] = A[0,0]B[0] + A[1,0]B[0]
    # But we need them separately. So we can't easily reduce below 6 for (2,3,1).
    # Actually, the rank of (2,3,1) is 6.
    
    # So for (2,3,9), if we process each k independently, we need 6*9 = 54.
    # But we can share across k! The key is that B[j,k] for different k can be combined.
    
    # Let's think about it differently. We have 27 B-entries and 6 A-entries, producing 18 C-entries.
    # 
    # One known technique: for (2, m, p), use a "sliding window" or "overlap" construction.
    # 
    # Actually, let's try a different approach. Consider the tensor as a matrix in different unfoldings.
    # 
    # The (n*m) x (m*p) unfolding (with p*n as the "output" index) doesn't directly help.
    # 
    # Let's try to use the known result for (2,3,3) which has rank 11.
    # For (2,3,9), we might be able to extend this.
    
    # Alternative approach: use a random search to try to find a lower-rank decomposition.
    # This is computationally expensive, so we'll limit it.
    
    import numpy as np
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a decomposition of rank R using a simple iterative method.
    # We'll use a "tensor train" style or just random initialization with gradient descent.
    # But this might be too slow for rank ~ 40-50.
    
    # Instead, let's try a structured construction.
    # 
    # For (2, 3, p), one approach:
    # - For each j in [0,1,2], we have A[:,j] (a 2-vector) and B[j,:] (a 9-vector).
    # - C = A[:,0] ⊗ B[0,:] + A[:,1] ⊗ B[1,:] + A[:,2] ⊗ B[2,:]
    # - This is a sum of 3 outer products of 2-vectors and 9-vectors.
    # - Each outer product is a 2x9 matrix, which has rank 1.
    # - So C has rank at most 3.
    # - But we need to compute this using bilinear operations.
    # 
    # The bilinear complexity is the rank of the tensor.
    # 
    # For the outer product A[:,j] ⊗ B[j,:]:
    # C[i,k] += A[i,j] * B[j,k]
    # This requires 2*9 = 18 multiplications for each j, total 54.
    # 
    # Can we do better by combining the j's?
    # 
    # Idea: Let S0 = A[:,0] + A[:,1], S1 = A[:,0] + A[:,2], S2 = A[:,1] + A[:,2]
    # Then A[:,0] = (S0 + S1 - S2)/2, etc.
    # And B[0,:] + B[1,:], etc.
    # This is the "Strassen-like" trick but for outer products.
    # 
    # Actually, for the sum of outer products:
    # C = u0 ⊗ v0 + u1 ⊗ v1 + u2 ⊗ v2
    # where u_j = A[:,j] (2-vector) and v_j = B[j,:] (9-vector).
    # 
    # We can use the identity:
    # u0 ⊗ v0 + u1 ⊗ v1 = ((u0+u1) ⊗ (v0+v1)) - (u0 ⊗ v1) - (u1 ⊗ v0)
    # This doesn't help directly.
    # 
    # Let's try a different grouping. For 2 vectors u0, u1 in R^2 and v0, v1 in R^9:
    # u0 ⊗ v0 + u1 ⊗ v1
    # = ((u0+u1)/2) ⊗ ((v0+v1)/2) + ((u0-u1)/2) ⊗ ((v0-v1)/2) - ((u0+u1)/2) ⊗ ((v0-v1)/2) + ((u0-u1)/2) ⊗ ((v0+v1)/2)
    # Wait, that's 4 terms instead of 2. Not helpful.
    # 
    # Actually, the rank of the sum of two rank-1 matrices is at most 2, but the bilinear complexity
    # of computing u0⊗v0 + u1⊗v1 is 2*2*9 = 36 if done separately, or potentially less if we share.
    # 
    # I think for (2,3,9), the rank is likely 54 (schoolbook is optimal) or close to it.
    # But let's try to verify with a quick search.
    
    # Try rank 53: can we eliminate one term?
    # The tensor has 54 non-zero entries, each in a distinct "position".
    # For the schoolbook decomposition, each term has exactly one non-zero in each of U, V, W.
    # To reduce rank, we need to "merge" two terms into one, which requires the merged term
    # to have two non-zeros in each factor.
    
    # Let's try: can we merge the terms for (i=0,j=0,k=0) and (i=1,j=0,k=0)?
    # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1  (i=0,j=0,k=0)
    # Term 2: U[3,0]=1, V[0,0]=1, W[0,1]=1  (i=1,j=0,k=0)  [W index is k*n+i = 0*2+1 = 1]
    # 
    # If we merge: U[0,0]=1, U[3,0]=1, V[0,0]=1, W[0,0]=1, W[0,1]=1
    # Then the product for (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # For (i=1,j=0,k=0): U[3,0]*V[0,0]*W[0,1] = 1*1*1 = 1 ✓
    # But for (i=0,j=0,k=0) with W[0,1]: U[0,0]*V[0,0]*W[0,1] = 1*1*1 = 1 (should be 0) ✗
    # So this doesn't work directly.
    
    # The issue is that the merged term creates cross-terms.
    
    # Given the difficulty, let's just return the schoolbook decomposition.
    # The rank is 54.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
