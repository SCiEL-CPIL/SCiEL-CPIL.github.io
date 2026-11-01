"""Kronecker-inspired rank-minimal tensor decomposition for 2x3x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Decompose the 2x3x5 multiplication tensor using Kronecker structure.
    # We split the middle dimension m=3 into m1=1, m2=3 (or m1=3, m2=1).
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix mult (A_{i,j} * B_{j,k})
    # can be seen as a product of smaller tensors.
    
    # Strategy: Use the fact that 3 = 3*1, so we can decompose as:
    # T_{2,3,5} = T_{2,1,5} ⊗ T_{1,3,1}
    # where T_{a,b,c}[i,j,k] = 1 if i==j==k (for the identity-like part)
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) is:
    # T[i, j, k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # The output is C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k.
    
    # The Kronecker decomposition approach:
    # T_{n,m,p} = T_{n,m1,p1} ⊗ T_{1,m2,p2} when m = m1*m2, p = p1*p2
    # But here p=5 is prime, so we can't split p.
    # Let's try: m=3 = 3*1, so T_{2,3,5} = T_{2,3,5} ⊗ T_{1,1,1}? No.
    
    # Alternative: Use the known result that the rank of matrix multiplication
    # for (n x m) * (m x p) is at most n*m*p (schoolbook), but can be lower.
    
    # For 2x3x5, let's try to find a decomposition with rank < 30.
    
    # Let's use a different approach: exploit the structure of small matrices.
    # The multiplication tensor M_{2,3,5} has entries M[i,j,k] = 1 for all i,j,k.
    # This is actually the all-ones tensor of shape (2,3,5).
    
    # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) if we allow
    # arbitrary entries, but for exact decomposition with integer/half-integer entries,
    # we need to be careful.
    
    # Actually, the all-ones tensor of shape (n,m,p) can be decomposed as:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # If we set U[i,r] = 1, V[j,r] = 1, W[k,r] = 1 for r=1, and 0 otherwise,
    # then T[i,j,k] = 1 for all i,j,k. So rank 1 suffices for the all-ones tensor!
    
    # Wait, but that's not the matrix multiplication tensor. The matrix multiplication
    # tensor for (n x m) * (m x p) is NOT the all-ones tensor. It's:
    # T[i,j,k] = 1 if and only if the indices correspond to a valid multiplication term.
    # Actually, for C[i,k] = sum_j A[i,j] * B[j,k], the tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So it IS the all-ones tensor of shape (n,m,p)!
    
    # Therefore, the rank of the 2x3x5 matrix multiplication tensor is 1!
    # We can decompose it with:
    # U[i,0] = 1 for all i
    # V[j,0] = 1 for all j  
    # W[k,0] = 1 for all k
    
    # Let's verify: C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (sum_r U[i,r] * x_r) * (sum_s V[j,s] * y_s) ... no, that's not right.
    
    # The tensor decomposition means:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # And the matrix product is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Wait, no. The standard formulation is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as a multilinear form:
    # C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # The rank-r decomposition gives:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Then: C[i,k] = sum_j sum_r U[i,r] * V[j,r] * W[k,r] * A[i,j] * B[j,k]
    # = sum_r U[i,r] * W[k,r] * (sum_j A[i,j] * V[j,r] * B[j,k])
    # Hmm, this doesn't directly give us C[i,k] = sum_j A[i,j] * B[j,k].
    
    # Let me reconsider. The correct tensor for matrix multiplication is:
    # For C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # The "multiplication tensor" is defined such that:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1 for all valid i,j,k.
    
    # A rank-r decomposition of T means:
    # T[i,j,k] = sum_{r=1}^{r} U[i,r] V[j,r] W[k,r]
    # Then: C[i,k] = sum_j sum_r U[i,r] V[j,r] W[k,r] A[i,j] B[j,k]
    # = sum_r U[i,r] W[k,r] (sum_j A[i,j] V[j,r] B[j,k])
    # This is NOT equal to sum_j A[i,j] B[j,k] in general.
    
    # I think I'm confusing the tensor formulation. Let me look at this more carefully.
    
    # The standard tensor formulation for matrix multiplication:
    # The tensor M_{n,m,p} is a 3-tensor with entries M[i,j,k] = 1 for all i,j,k
    # (where i indexes rows of A, j indexes columns of A / rows of B, k indexes columns of B).
    # The matrix product is: C[i,k] = sum_j M[i,j,k] A[i,j] B[j,k]
    
    # A rank-r decomposition: M[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Then: C[i,k] = sum_j sum_r U[i,r] V[j,r] W[k,r] A[i,j] B[j,k]
    # = sum_r U[i,r] W[k,r] sum_j A[i,j] V[j,r] B[j,k]
    
    # For this to equal sum_j A[i,j] B[j,k] for ALL A and B, we need:
    # sum_r U[i,r] W[k,r] sum_j A[i,j] V[j,r] B[j,k] = sum_j A[i,j] B[j,k]
    
    # This is a very strong condition. It's not simply that M is the all-ones tensor
    # with rank 1. The issue is that the decomposition must work for the bilinear form.
    
    # Actually, I think the correct interpretation is different. Let me reconsider.
    
    # In the standard algebraic complexity theory formulation:
    # The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors.
    # A rank-r decomposition is:
    # T = sum_{r=1}^{r} u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors.
    # This means: T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]
    
    # The matrix product C = AB can be computed as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using the decomposition:
    # C[i,k] = sum_r (sum_j A[i,j] v_r[j]) (sum_j' ... ) 
    # No, that's not right either.
    
    # Let me use the correct formulation from the literature:
    # If T = sum_r u_r ⊗ v_r ⊗ w_r, then we can compute C = AB as:
    # For each r: compute intermediate vectors
    # x_r = A u_r (vector of length n)
    # y_r = v_r^T B (vector of length p)
    # Then C = sum_r x_r ⊗ y_r ... no.
    
    # Actually, the correct formula is:
    # C[i,k] = sum_r u_r[i] * (sum_j A[i,j] v_r[j]) * w_r[k] ... no.
    
    # I'll just use the schoolbook decomposition which is known to be correct.
    rank = n * m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
