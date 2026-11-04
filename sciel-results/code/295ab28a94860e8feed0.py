"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    # Schoolbook decomposition: rank = n*m*p = 100
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using bilinear search
    # We'll use a simple approach: try to find rank-1 terms that cover the tensor
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} * (A_{i,j} * B_{j,k})
    # Actually, the tensor is T[i,j,k] = 1 if we're looking at the structure where
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the sense that each term A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 (it's the structure tensor where each trilinear term is A[i,j]*B[j,k]*C[i,k])
    
    # Let's try a simple rank reduction: look for linear dependencies
    # Use a greedy approach: try to express the tensor as sum of rank-1 tensors
    # A rank-1 tensor is u[i]*v[j]*w[k]
    
    # For matrix multiplication, the tensor T[i,j,k] = 1 if we think of it as
    # the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p) where the multiplication is defined
    # Actually, the standard matrix multiplication tensor has:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is indexed by (i,j,k) and T[i,j,k] = 1
    
    # Let's build the full tensor and try to find a low-rank decomposition
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Actually, I need to be more careful. The matrix multiplication tensor for n x m by m x p is:
    # It's a tensor of size n x m x p where T[i,j,k] = 1
    # But that's not right either. Let me think again.
    
    # The bilinear form is: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears exactly once in C[i,k])
    
    # Wait, that's not right. The tensor should be:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Which is always true, so T[i,j,k] = 1 for all valid indices
    
    # But that would mean the tensor is all-ones, which has rank 1, which is clearly wrong.
    
    # Let me reconsider. The matrix multiplication tensor is:
    # T[i,j,k] where i indexes rows of A (and C), j indexes columns of A (and rows of B), k indexes columns of B (and C)
    # The tensor satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: C[i,k] = sum_{j,l} T[i,j,l] * A[i,j] * B[l,k]
    # So T[i,j,l] = delta_{j,l}
    
    # Ah, so the tensor is T[i,j,k] = delta_{j,k} where j is the column of A / row of B, and k is the column of B
    # Wait, I'm confusing myself. Let me use standard notation.
    
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The tensor T has dimensions n x m x p (or in the factorization format, the "middle" index is m)
    # T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # Since each (i,j,k) triple uniquely identifies one term in one C entry, T[i,j,k] = 1 for all i,j,k
    
    # But this is the "structure" tensor. The actual tensor to decompose is different.
    
    # In the context of bilinear complexity, the tensor for matrix multiplication is:
    # T in V^* ⊗ W^* ⊗ Z^* where V, W, Z are the spaces for A, B, C respectively
    # T = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ f_k^* where the sum is over all (i,j,k) such that
    # the product A[i,j] * B[j,k] appears in C[i,k]
    
    # Actually, I think the correct tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ f_k
    # where e_i is the i-th basis vector of R^n, e_j is the j-th basis vector of R^m, f_k is the k-th basis vector of R^p
    # And the bilinear form is <T, (A,B,C)> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    
    # For matrix multiplication, we want: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor should satisfy: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = 0 for all A,B where C = A*B
    # Or rather, the tensor defines the bilinear map, and we want to decompose it.
    
    # I think I'm overcomplicating this. Let me just use the schoolbook decomposition which is known to be valid.
    # The score is 1.0 for rank 100, which is the schoolbook rank.
    
    # Let me try a different approach: use the known fact that for 4x5x5,
    # we can potentially do better than 100.
    
    # Actually, let me just return the schoolbook decomposition since it's guaranteed valid.
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
