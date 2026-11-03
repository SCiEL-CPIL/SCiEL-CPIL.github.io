"""Program for matrix multiplication 3x4x4 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Use a rank-16 decomposition based on the following approach:
    # For the 4x4x4 tensor, we can use a decomposition of rank 16
    # by exploiting the structure. For 3x4x4, we can potentially do better.
    
    # Let's use a systematic approach:
    # The tensor T[i,j,k] = 1 for the matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to find rank-r decomposition.
    
    # One approach: use the fact that we can combine the i and j dimensions.
    # For each k, we have a 3x4 matrix of A values and 4x1 vector of B values.
    
    # Let's try rank 16:
    # For each of the 4 values of k, we need to compute C[:,k] = A * B[:,k]
    # This is a 3x4 matrix times a 4x1 vector, giving a 3x1 vector.
    # The rank of this operation for a single k is 4 (each output depends on all 4 inputs).
    # But we can share structure across k.
    
    # Actually, let's use a known approach: 
    # The rank of the 4x4x4 matrix multiplication tensor is at most 15 (Winograd),
    # but for 3x4x4, we might be able to do better.
    
    # Let me construct a rank-16 decomposition:
    # For each j in 0..3 and each k in 0..3, we have a "product" A[i,j]*B[j,k]
    # But we can combine some of these.
    
    # Simple approach: rank = n*m*p = 48 (schoolbook)
    # Better: rank = m*p = 16 if we can do it per j
    
    # Let's try: for each j (0..3), compute all i and k at once.
    # A[i,j] is a 3x1 vector (for fixed j), B[j,k] is a 1x4 vector (for fixed j)
    # Their outer product is 3x4, contributing to C.
    # So for each j, we need rank 1 tensor? No, the outer product of a 3-vector and 4-vector
    # has rank 1 in the (i,k) space, but we're summing over j.
    
    # Actually: C = sum_j (A[:,j] outer B[j,:])
    # Each term A[:,j] outer B[j,:] is a rank-1 matrix in the (i,k) space.
    # So the total rank in the (i,k) space is at most 4 (sum of 4 rank-1 matrices).
    # But this is not the tensor rank we need.
    
    # The tensor rank of 3x4x4 matrix multiplication:
    # T[i,j,k] = delta_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k with T[i,j,k]=1 for the "diagonal" 
    # in the appropriate sense.
    
    # Actually, for matrix multiplication, T[i,j,k] = 1 if we're computing the (i,k) entry
    # from A[i,j] and B[j,k]. So T[i,j,k] = 1 for all i,j,k (it's the full tensor where
    # each slice T[:,:,k] is the identity-like structure).
    
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Because C[i,k] = sum_j A[i,j] * B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k]
    # is 1 for all i,j,k.
    
    # So T is the 3x4x4 all-ones tensor? No, that's not right either.
    # T[i,j,k] = 1 means that A[i,j]*B[j,k] contributes to C[i,k] with weight 1.
    # Yes, so T is indeed the all-ones tensor of shape (3,4,4).
    
    # The rank of the all-ones tensor of shape (n,m,p) is min(n,m,p) if we allow
    # arbitrary entries, but for the matrix multiplication tensor, it's different.
    
    # Actually, I think I'm confusing myself. Let me just use the schoolbook approach
    # but try to optimize.
    
    # Schoolbook: rank = n*m*p = 3*4*4 = 48
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
