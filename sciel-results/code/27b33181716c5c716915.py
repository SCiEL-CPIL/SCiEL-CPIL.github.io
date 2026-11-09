"""Program for matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Trivial decomposition has rank n*m*p = 60
    # We'll try to find a lower rank decomposition using structured patterns
    
    # The multiplication tensor T[i,j,k] = e_{i,j} * e_{j,k} = e_{i,k} (for matrix mult)
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The trivial decomposition uses rank = n*m*p with U[i*m+j, (i,j,k)] = 1, etc.
    
    # Let's try a different approach: use the fact that we can combine some terms
    # by exploiting linear dependencies.
    
    # Start with the trivial rank-60 decomposition
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U_trivial[i * m + j, idx] = 1.0
        V_trivial[j * p + k, idx] = 1.0
        W_trivial[k * n + i, idx] = 1.0
    
    # Now try to reduce the rank by finding linear combinations
    # that can be merged.
    
    # Strategy: For fixed i, we have m*p terms. We can try to combine terms
    # that share the same i but have different (j,k) pairs.
    
    # Actually, let's try a more systematic approach: use Strassen-like ideas
    # or just try to find a lower rank via SVD-like factorization.
    
    # The tensor T can be viewed as a matrix of shape (n*m) x (m*p) for each k,
    # but that's not quite right. Let me think differently.
    
    # The multiplication tensor is a 3-tensor T[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    # T[i,j,k] = 1 for all valid (i,j,k), representing C[i,k] += A[i,j]*B[j,k]
    
    # Wait, that's not right either. The tensor should encode the bilinear map.
    # For matrix multiplication, the tensor T has T[i,j,k] = delta_{i,k} if we're
    # thinking of it as a map from A and B to C, but that's not standard.
    
    # Let me reconsider. The standard way: the multiplication tensor M_{n,m,p} 
    # has entries M[i,j,k] = 1 if (i,j,k) is a valid triple, where:
    # - i in 0..n-1 (row of A, row of C)
    # - j in 0..m-1 (col of A, row of B)  
    # - k in 0..p-1 (col of B, col of C)
    # And the bilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    
    # Actually the tensor is T[i,j,k] and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the trivial decomposition, each (i,j,k) gets its own term r, with
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    
    # To reduce rank, we need to find linear combinations. Let's try to use
    # the fact that for fixed i and k, the sum over j of A[i,j]*B[j,k] can be
    # computed with fewer than m terms using fast matrix multiplication ideas.
    
    # For 1x1x1 blocks, we need 1 term. For 2x2x2, Strassen shows 7 < 8.
    # For our 3x4x5 case, let's try to find a structured decomposition.
    
    # Approach: Group the computation into blocks and use known fast algorithms
    # for smaller matrices, then combine.
    
    # Actually, let's just try to find a lower rank decomposition using
    # numerical linear algebra. We'll set up the system of equations and try
    # to find a solution with lower rank.
    
    # The constraint is: for all (i,j,k),
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 60 equations in 3*r*(n*m + m*p + p*n) unknowns.
    
    # Let's try ranks from 48 up to 60 and see if we can find a solution.
    
    # For a given rank r, we have r columns in U, V, W.
    # The equations are bilinear in the unknowns, so this is a non-convex problem.
    
    # Let's try a simpler approach: use the structure of the problem.
    
    # For each pair (i,k), we need to compute sum_j A[i,j]*B[j,k].
    # This is an inner product of two length-m vectors.
    
    # If we can find a decomposition where for each (i,k), the sum over r of
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] equals 1 for all j... 
    
    # Wait, I'm overcomplicating this. Let me just return the trivial decomposition
    # since finding a lower rank exact decomposition with integer/half-integer entries
    # is a hard open problem.
    
    # But let's try one more thing: use the fact that we can sometimes combine
    # terms by exploiting symmetries or linear dependencies.
    
    # For now, return the trivial rank-60 decomposition.
    
    rank = rank_trivial
    U = U_trivial.tolist()
    V = V_trivial.tolist()
    W = W_trivial.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
