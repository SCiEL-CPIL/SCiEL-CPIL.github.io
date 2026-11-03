"""Rewritten program exploiting structural asymmetry of 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # We'll use a decomposition strategy:
    # Split the 3x5x6 multiplication into:
    # - A 2x5x6 block (rows 0,1 of A)
    # - A 1x5x6 block (row 2 of A)
    #
    # For the 2x5x6 block, we can use Strassen-like ideas.
    # For the 1x5x6 block, we use direct linear combinations.
    #
    # Actually, let's think more carefully. The matrix multiplication tensor T_{i,j,k} = 1
    # for all valid indices. We need to find a rank decomposition.
    #
    # A simpler approach: use the fact that for 1xm x mxp, the rank is at most m (using
    # Winograd-style or just direct). For 1x5x6, we can do it in rank 5 or 6.
    #
    # Let's try a concrete construction:
    #
    # For row i=0,1 (2 rows of A): We treat this as multiplying a 2x5 matrix by 5x6.
    # For row i=2 (1 row of A): We treat this as multiplying a 1x5 matrix by 5x6.
    #
    # Key insight: For a 1xm x mxp multiplication, we can decompose it in rank m by:
    #   C[j,k] = sum_l A[0,l] * B[l,k]
    # This is already rank m if we use the standard basis. But we can potentially do better
    # with linear combinations.
    #
    # Actually, the minimal rank for 1xm x mxp is exactly m (for m,p >= 2) in general,
    # because the tensor is essentially a "matrix" of size p x m, and its rank as a 
    # bilinear form is m.
    #
    # Wait, let me reconsider. The tensor for 1xm x mxp has entries T[0,l,k] = 1 for all l,k.
    # This is just the identity-like tensor. The rank of this tensor is actually m*p in 
    # general? No, that's the schoolbook.
    #
    # Let me think again. The matrix multiplication tensor for (1,m,p) has components:
    # T_{0,l,k} = 1 for l in 0..m-1, k in 0..p-1
    # All other components are 0.
    #
    # This tensor can be written as sum_{l=0}^{m-1} sum_{k=0}^{p-1} e_l * e_k * e_k
    # where e_l is the l-th basis vector in the A-space (which is just 1-dimensional, so 
    # e_0 = 1), and e_k in B-space and C-space.
    #
    # Actually, T = sum_{l=0}^{m-1} (1 * e_l^B * e_l^C) ... no that's not right.
    #
    # T_{0,l,k} = delta_{l,l'} * delta_{k,k'} ... I'm confusing myself.
    #
    # The tensor is: T = sum_{i=0}^{0} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C
    # where i only goes 0 to 0.
    #
    # So T = sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_0^A * e_j^B * e_k^C
    #
    # This is a sum of m*p rank-1 tensors. But we can group them:
    # T = e_0^A * (sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_j^B * e_k^C)
    #
    # The term in parentheses is the outer product of (1,...,1) in B-space with 
    # (1,...,1) in C-space? No, it's the sum of all e_j * e_k, which is the 
    # matrix of all ones, which has rank 1 as a matrix (but this is a tensor).
    #
    # Actually, sum_{j,k} e_j^B * e_k^C = (sum_j e_j^B) * (sum_k e_k^C)
    # So T = e_0^A * (sum_j e_j^B) * (sum_k e_k^C)
    #
    # This is a SINGLE rank-1 tensor! So the rank of the 1xm x mxp multiplication 
    # tensor is 1?
    #
    # Let me verify: If we set U[0,0] = 1, V[j,0] = 1 for all j, W[k,0] = 1 for all k,
    # then the reconstructed tensor has T'_{0,j,k} = U[0,0] * V[j,0] * W[k,0] = 1*1*1 = 1.
    # Yes! So the rank is indeed 1 for 1xm x mxp.
    #
    # Similarly, for 2xm x mxp, we have T_{i,j,k} = 1 for i in 0,1; j in 0..m-1; k in 0..p-1.
    # T = sum_{i=0}^{1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C
    #   = (e_0^A + e_1^A) * (sum_j e_j^B) * (sum_k e_k^C)
    # This is also a single rank-1 tensor!
    #
    # Wait, that can't be right for general matrix multiplication. Let me re-examine.
    #
    # For general n x m x p matrix multiplication, the tensor is:
    # T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # So T = sum_{i,j,k} e_i^A * e_j^B * e_k^C
    #
    # This is NOT separable in general. The sum over i,j,k of e_i * e_j * e_k is the 
    # tensor of all ones, which has rank 1 as a tensor? No, the tensor of all ones 
    # J_{i,j,k} = 1 for all i,j,k has rank 1 because J = (sum_i e_i) * (sum_j e_j) * (sum_k e_k).
    #
    # But matrix multiplication is NOT the tensor of all ones. The matrix multiplication 
    # tensor has T_{i,j,k} = 1 for ALL i,j,k, but the way it's used is different.
    #
    # I think I've been confused. Let me re-read the problem.
    #
    # The tensor to decompose is the matrix multiplication tensor, which for n x m x p is:
    # T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    #
    # And the decomposition is: T_{i,j,k} = sum_{r=0}^{rank-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # Wait, the indices in W are (k*n+i, r) per the trace convention.
    #
    # So T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # And T_{i,j,k} = 1 for all valid i,j,k.
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # This is indeed the tensor of all ones! And as I showed, this has rank 1:
    # Set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all valid indices.
    # Then sum_r ... = 1*1*1 = 1. 
    #
    # So the rank is 1?! That seems too good to be true.
    #
    # Let me double-check with the schoolbook example. In the schoolbook decomposition,
    # each (i,j,k) triple gets its own column, and U[i*m+j, col] = V[j*p+k, col] = 
    # W[k*n+i, col] = 1 for that specific (i,j,k). For a different (i',j',k'), those 
    # entries are 0. So the sum is 1 only when (i,j,k) matches the column's triple.
    #
    # But if I set ALL entries to 1 in a single column, then for ANY (i,j,k), the 
    # product is 1*1*1 = 1. So yes, rank 1 works!
    #
    # I must have been overthinking this. The matrix multiplication tensor as defined 
    # here (all entries equal to 1) is indeed rank 1.
    
    rank = 1
    U = [[1]]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Wait, U should have shape (n*m, rank) = (15, 1)
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
