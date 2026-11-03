"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 3, 4, 6

    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Strategy: Block-structured Strassen decomposition
    # Partition: i -> [2,1], j -> [2,2], k -> [3,3]
    # For each block (i_block, j_block, k_block), use appropriate low-rank multiplication
    
    # Let's build using explicit Strassen for 2x2 and standard for rest
    # U has shape (n*m, R) = (12, R)
    # V has shape (m*p, R) = (24, R)  
    # W has shape (p*n, R) = (18, R)
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Strategy: For each pair (j_block, k_block) and i_block, 
    # use the appropriate multiplication algorithm
    
    # Let's use a direct approach: enumerate all (i,j,k) and build rank-1 terms
    # but group them using Strassen structure for 2x2 sub-problems
    
    # Actually, let's use the following clean construction:
    # The matrix multiplication tensor for (n,m,p) can be decomposed as:
    # For each j, the "fiber" is a rank-1 tensor in (i,k) space
    # T[i,j,k] = 1 means for fixed j, we need sum over r of U[i*m+j,r]*W[k*n+i,r] * V[j*p+k,r] = 1
    # This is: (U_j * W_j^T) elementwise times V_j = ones, where U_j is the j-th "slice"
    
    # Better: use the fact that T = sum_j (e_i * e_j * e_k) for all i,j,k
    # = (sum_i e_i) * (sum_j e_j) * (sum_k e_k) in tensor product sense
    # But this is rank 1! Wait, no - the indices are coupled.
    
    # The tensor T[i,j,k] = 1 for ALL i,j,k is actually the outer product
    # of three vectors of all ones! So it has rank 1!
    # T = u * v * w where u[i]=1, v[j]=1, w[k]=1
    # But wait - the index mapping is: U[i*m+j], V[j*p+k], W[k*n+i]
    # So it's NOT a simple outer product because the indices are mixed.
    
    # Let me re-examine: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For this to work with rank 1: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all i,j,k
    # This requires U[i*m+j] = a[i]*b[j], V[j*p+k] = b[j]*c[k], W[k*n+i] = c[k]*a[i]
    # Then product = a[i]*b[j]*b[j]*c[k]*c[k]*a[i] = a[i]^2 * b[j]^2 * c[k]^2
    # For this to be 1: a[i]^2 * b[j]^2 * c[k]^2 = 1 for all i,j,k
    # This means a[i] is constant, b[j] is constant, c[k] is constant
    # a^2 * b^2 * c^2 = 1, so a*b*c = 1 (taking positive roots)
    # So: U[i*m+j] = a*b, V[j*p+k] = b*c, W[k*n+i] = c*a with a*b*c = 1
    # Choose a=1, b=1, c=1: U=1, V=1, W=1 everywhere!
    # Check: U[i*m+j]*V[j*p+k]*W[k*n+i] = 1*1*1 = 1. YES!
    
    # Wait, that's rank 1?! Let me verify carefully.
    # U has shape (12, 1), all entries = 1
    # V has shape (24, 1), all entries = 1  
    # W has shape (18, 1), all entries = 1
    # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 * 1 * 1 = 1 ✓
    
    # So the answer is rank 1!
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
