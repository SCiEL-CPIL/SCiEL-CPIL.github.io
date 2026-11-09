import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # 
    # This is equivalent to: for each (i,j,k), the vector u_{i,j} = U[i*m+j, :]
    # v_{j,k} = V[j*p+k, :] and w_{k,i} = W[k*n+i, :] must satisfy
    # u_{i,j} . v_{j,k} . w_{k,i} = 1
    #
    # Actually the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # The schoolbook algorithm uses rank n*m*p = 90.
    # 
    # For the specific case of 3x5x6, let's think about whether we can do better.
    # The key insight is that we need a set of vectors {u_a, v_b, w_c} such that
    # for each triple (i,j,k), there exists some combination that gives 1.
    #
    # Actually, let me reconsider the problem. The tensor is:
    # T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,6)
    #
    # We want to write T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is a rank-r decomposition of the all-ones tensor of shape (3*5, 5*6, 6*3) = (15, 30, 18).
    #
    # The all-ones tensor of shape (a, b, c) has rank min(a, b, c) if we allow real entries?
    # No, that's not right either. The all-ones tensor has rank 1!
    #
    # Wait: T[i,j,k] = 1 for all i,j,k. This can be written as:
    # T[i,j,k] = u_i * v_j * w_k where u_i = 1, v_j = 1, w_k = 1
    #
    # But our indexing is different. Let me re-examine.
    #
    # The multiplication tensor for (n,m,p) is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1. The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So we need vectors indexed by (i,j), (j,k), (k,i) such that their inner product sum is 1.
    #
    # Key observation: If we set U[i*m+j, r] = a_{i,j,r}, V[j*p+k, r] = b_{j,k,r}, W[k*n+i, r] = c_{k,i,r}
    # then we need sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1 for all i,j,k.
    #
    # One simple solution: use r = 1, with a_{i,j,0} = 1, b_{j,k,0} = 1, c_{k,i,0} = 1.
    # Then sum_r = 1*1*1 = 1. This gives rank 1!
    #
    # But wait, is this valid? Let me check with the actual matrix multiplication.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # 
    # The tensor T[i,j,k] = 1 means that the coefficient of A_{i,j} * B_{j,k} in (AB)_{i,k} is 1.
    # So (AB)_{i,k} = sum_j T[i,j,k] * A_{i,j} * B_{j,k} = sum_j A_{i,j} B_{j,k}.
    #
    # With rank 1 decomposition: T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1.
    # This is correct!
    #
    # So the rank is 1? That seems too good to be true. Let me double-check.
    #
    # Actually, I think I'm confusing the tensor decomposition with the algorithmic complexity.
    # In tensor decomposition for matrix multiplication, the rank of the tensor corresponds to
    # the number of scalar multiplications needed. The all-ones tensor of shape (n, m, p) 
    # does have tensor rank 1 because it's a rank-1 tensor (outer product of three all-ones vectors).
    #
    # But this doesn't make sense for matrix multiplication. The issue is that the tensor
    # T[i,j,k] = 1 is indeed rank 1 as a 3-tensor, but the matrix multiplication algorithm
    # requires more structure.
    #
    # Let me reconsider. The standard result is that the matrix multiplication tensor 
    # M_{n,m,p} has tensor rank at least max(nm, mp, np) and at most nmp.
    #
    # For n=3, m=5, p=6: lower bound = max(15, 30, 18) = 30.
    #
    # So rank 1 is definitely wrong. The error in my reasoning is that while T[i,j,k]=1 
    # as a 3-tensor is rank 1, the way it's embedded in the matrix multiplication problem
    # with the specific index mappings U[i*m+j], V[j*p+k], W[k*n+i] imposes additional constraints.
    #
    # Actually no, the tensor IS the matrix multiplication tensor. Let me look at this more carefully.
    #
    # The matrix multiplication tensor is defined as:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # where e_{i,j} is the basis vector in R^{n*m} with 1 at position i*m+j, etc.
    #
    # This tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    #
    # So it's NOT the all-ones tensor of shape (nm, mp, np). It's a sparse tensor where
    # only specific entries are 1.
    #
    # The rank of this tensor is what we're looking for. For n=3, m=5, p=6, the known
    # upper bound is 90 (schoolbook) and lower bound is 30.
    #
    # Let me try a different approach. Can we find a decomposition with rank < 90?
    #
    # One idea: use the fact that 5 and 6 are composite. But 5 is prime.
    #
    # Another idea: use block decomposition. Split the 5x6 matrix into blocks.
    # For example, split the 5 rows into groups and the 6 columns into groups.
    #
    # Actually, let me just try to find a rank-30 decomposition if possible, or at least
    # something better than 90.
    #
    # For now, let me use a simple approach: try to find a decomposition using random
    # vectors and check if it works. But that's unlikely to find the minimum.
    #
    # Let me just return the schoolbook decomposition with rank 90, which is known to be correct.
    
    rank = n * m * p
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
