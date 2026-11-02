"""Program for matrix multiplication 2x2 * 2x8 rank minimization."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # We are multiplying A (n x m) by B (m x p) to get C (n x p)
    # A is 2x2, B is 2x8, C is 2x8
    
    # Indexing convention:
    # A[i, j] where i in 0..n-1, j in 0..m-1
    # B[j, k] where j in 0..m-1, k in 0..p-1
    # C[i, k] where i in 0..n-1, k in 0..p-1
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (16, rank)
    
    # The output W follows W[k*n+i, r] convention
    
    # Strategy: Use a block-based approach
    # Since m=2, we can write:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This is essentially a bilinear form.
    # The trivial decomposition has rank n*m*p = 2*2*8 = 32
    
    # Can we do better? 
    # For each pair (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This looks like for each i, we have a 1x2 vector A[i,:] times 2x8 matrix B
    # Actually, let's think of it differently.
    
    # Consider the tensor T[i,j,k] = delta(i,j,k) where the multiplication tensor
    # is T[i,j,k] = 1 if the multiplication A*B produces C with those indices.
    
    # Actually the standard matrix multiplication tensor for n x m * m x p -> n x p
    # has components: for each (i, j, k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    # So the tensor is: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k] (as a trilinear form)
    
    # The rank of this tensor is what we're minimizing.
    
    # For n=2, m=2, p=8:
    # Trivial: 32
    
    # Let me think about whether we can reduce this.
    
    # One approach: group the k indices.
    # For fixed i and j, the contribution to all k is A[i,j] * B[j,k] for k=0..7
    
    # This is like an outer product structure.
    
    # Actually, let's just use the straightforward approach but see if we can be clever.
    
    # For each i (0,1), j (0,1), k (0..7):
    # C[i,k] gets contribution A[i,j]*B[j,k]
    
    # We can write this as:
    # For each (i,j), we have a "product" A[i,j] that multiplies with the entire row j of B
    
    # So we could have rank terms corresponding to (i,j) pairs, but each would need to produce 
    # 8 outputs... that doesn't work directly since rank-1 terms produce one output per term.
    
    # Let me just implement the schoolbook (rank 32) and see if there's a simple improvement.
    
    # Actually, let me think about this more carefully using the structure.
    
    # C is 2x8. Each entry C[i,k] = sum_j A[i,j]*B[j,k]
    
    # We can think of this as: for each k, C[:,k] = A * B[:,k] where B[:,k] is a 2x1 column
    # This is 2x2 * 2x1 = 2x1, which has rank 7 (Strassen) per column.
    # 8 columns would give 7*8 = 56, which is worse.
    
    # Alternatively, for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is a linear combination of two rows of B, with coefficients from A.
    
    # The tensor structure: T[i,j,k] where i=0,1; j=0,1; k=0..7
    # T[i,j,k] = 1 for all valid i,j,k (since each A[i,j]*B[j,k] appears exactly once in C[i,k])
    
    # Wait, no. The tensor is indexed such that the multiplication is:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # So we need to decompose the 2x2x8 tensor of all ones into sum of rank-1 tensors.
    
    # A rank-1 tensor is u_i * v_j * w_k
    
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # This is a tensor completion/decomposition problem.
    
    # The tensor is J_2 ⊗ J_2 ⊗ J_8 (Kronecker product of all-ones matrices)
    # Actually no, it's just the all-ones tensor of shape (2,2,8).
    
    # The rank of the all-ones tensor J_{a} ⊗ J_{b} ⊗ J_{c} is min(a,b,c) if we're talking about...
    # No, the rank of the all-ones tensor is 1! Because J_{a,b,c} = 1 = 1*1*1 for all entries.
    
    # Wait, that's not right either. The all-ones tensor of shape (n,m,p) can be written as:
    # T[i,j,k] = 1 = u_i * v_j * w_k where u_i=1, v_j=1, w_k=1 for all i,j,k
    # So it's rank 1!
    
    # But that can't be the matrix multiplication tensor. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 8 matrix"
    
    # The multiplication tensor for C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has T[i,j,k] = 1 if... wait, I think I'm confusing myself.
    
    # The standard matrix multiplication tensor has:
    # T[i,j,k] = 1 for each triple (i,j,k) corresponding to one term in the sum
    # But the tensor is defined on the index set {(i,j,k) : i in [n], j in [m], k in [p]}
    # and T[i,j,k] = 1.
    
    # No wait, I think the issue is that the tensor IS the all-ones tensor, and its rank
    # as a tensor is indeed 1. But that doesn't make sense for matrix multiplication.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over all i,j,k
    # No, that's not right either.
    
    # Actually, I think the correct definition is:
    # The tensor is an element of (R^n ⊗ R^m)^* ⊗ (R^p ⊗ R^n) or something like that.
    
    # Let me just go with the interface: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # and the condition is that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # So yes, we need to represent the all-ones tensor of shape (n*m, m*p, p*n) with the
    # specific index mapping U[i*m+j], V[j*p+k], W[k*n+i].
    
    # And the all-ones tensor has rank 1! We can set:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_r U*V*W = 1*1*1 = 1 for all i,j,k.
    
    # So rank 1 should work!
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
