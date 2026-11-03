"""Program for matrix multiplication 2x3x6 tensor decomposition."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    The tensor T has indices (i, j, k) where:
    - i in [0, 1] (row of A, size n=2)
    - j in [0, 2] (column of A / row of B, size m=3)  
    - k in [0, 5] (column of B, size p=6)
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    The factor matrices have shapes:
    - U: (n*m, rank) = (6, rank)
    - V: (m*p, rank) = (18, rank)
    - W: (p*n, rank) = (12, rank)
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, let's try to find a low-rank decomposition.
    # The tensor has 36 non-zero entries (one for each (i,j,k) triple).
    # The trivial rank is 36 (one term per triple).
    # We can potentially do better by sharing structure.
    
    # Key insight: For matrix multiplication, the tensor is "structured".
    # Specifically, T[i,j,k] = 1 if the indices are consistent (which they always are
    # in our indexing), so we need to decompose the identity-like tensor.
    
    # Actually, the matrix multiplication tensor for (n,m,p) has entries:
    # T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, that's not quite right. Let me reconsider.
    
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # M(i,j,k) = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So M(i,j,k) = 1 for all i in [0,n-1], j in [0,m-1], k in [0,p-1]
    # This is just the all-ones tensor of size n*m*p!
    
    # Wait, that can't be right either. Let me think more carefully.
    
    # In the standard formulation, the matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta(j, j') * delta(k, k') * delta(i, i')
    # No, that's not right.
    
    # Let me use the standard definition. The matrix multiplication tensor for
    # multiplying an n×m matrix by an m×p matrix is a 3-tensor T of size (n*m) × (m*p) × (p*n)
    # such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # No wait, that's for the identity.
    
    # Actually, I think the correct formulation is:
    # We want to compute C = A * B where A is n×m and B is m×p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition gives:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    # Hmm, this is getting confused.
    
    # Let me use the standard Strassen formulation:
    # We want: sum_c M_c(i,j,k) = T(i,j,k) where T(i,j,k) = 1 for all valid indices
    # and M_c(i,j,k) = U_c(i,j) * V_c(j,k) * W_c(k,i)
    
    # Actually, I think the simplest way to think about it:
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p (trivial)
    # and at least max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    
    # For (2,3,6), let's try to find a decomposition of rank 18 or lower.
    
    # Let me try a systematic approach: use Gaussian elimination over the rationals
    # to find the minimal rank.
    
    # Actually, let me first try a constructive approach.
    
    # For n=2, m=3, p=6:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We can group terms by k. For each k, we have 6 products (3 for row 0, 3 for row 1).
    # Total: 6*6 = 36 products.
    
    # Can we share? Note that B[j,k] appears in both C[0,k] and C[1,k].
    # So for fixed (j,k), the term A[0,j]*B[j,k] and A[1,j]*B[j,k] share B[j,k].
    
    # One idea: use 2 terms per (j,k) pair, one for each i. That's still 2*3*6 = 36.
    
    # Better idea: for fixed (j,k), use one term that contributes to both i=0 and i=1.
    # Term: U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=1
    # This gives: contributes B[j,k] to both C[0,k] and C[1,k].
    # But we need A[0,j]*B[j,k] for C[0,k] and A[1,j]*B[j,k] for C[1,k].
    # With this term, we get B[j,k] for both, not A[0,j]*B[j,k] and A[1,j]*B[j,k].
    
    # So that doesn't work directly. We need the U values to encode A[i,j].
    
    # Let me try: for each (j,k), use 2 terms:
    # Term 1: U[0*m+j]=1, U[1*m+j]=0, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=0
    #   -> contributes A[0,j]*B[j,k] to C[0,k]
    # Term 2: U[0*m+j]=0, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=0, W[k*n+1]=1
    #   -> contributes A[1,j]*B[j,k] to C[1,k]
    # Total: 2*3*6 = 36 terms.
    
    # To reduce, we need to combine terms across different (j,k) pairs.
    
    # Let me try a different approach: use the fact that n=2 is small.
    # We can use 2 "basis" vectors for the i dimension.
    
    # Idea: For each k, use 3 terms (one per j) for i=0, and 3 terms for i=1.
    # But share the V and W factors where possible.
    
    # Actually, let me just try to find a rank-18 decomposition by solving a linear system.
    
    # The tensor has 36 entries (one for each (i,j,k)).
    # Each rank-1 term is a product of three vectors.
    # We need to find r rank-1 terms that sum to the tensor.
    
    # This is a nonlinear problem, but we can try to solve it iteratively.
    
    # For now, let me just return the trivial rank-36 decomposition and see if
    # we can improve it.
    
    # Actually, let me try a smarter approach. Let me use the following construction:
    
    # For each k in 0..5, define 3 terms that handle all 6 products for that k.
    # Term (k, j=0): U[0]=1, U[3]=0, V[0]=1, V[6]=0, V[12]=0, W[k]=1, W[k+6]=1
    #   -> contributes A[0,0]*B[0,k] to C[0,k] and A[1,0]*B[0,k] to C[1,k]
    # Wait, that's not right. Let me be more careful.
    
    # U has indices (i*m+j) for i in 0..1, j in 0..2. So U[0]=U[i=0,j=0], U[1]=U[i=0,j=1], U[2]=U[i=0,j=2], U[3]=U[i=1,j=0], U[4]=U[i=1,j=1], U[5]=U[i=1,j=2].
    # V has indices (j*p+k) for j in 0..2, k in 0..5. So V[0]=V[j=0,k=0], ..., V[17]=V[j=2,k=5].
    # W has indices (k*n+i) for k in 0..5, i in 0..1. So W[0]=W[k=0,i=0], W[1]=W[k=0,i=1], W[2]=W[k=1,i=0], W[3]=W[k=1,i=1], ..., W[11]=W[k=5,i=1].
    
    # For a rank-1 term c, the contribution to C[i,k] is:
    # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want this to equal 1 for all (i,j,k) when summed over c, but that's not right.
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    # Wait, that's still not right. The tensor entry T(i,j,k) should be 1 for all valid (i,j,k).
    
    # Let me re-read the problem statement:
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # I think the tensor is: T(i,j,k) = 1 for all i in 0..1, j in 0..2, k in 0..5.
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T(i,j,k) = 1.
    
    # So we need to decompose the all-ones tensor of size 2*3*6 = 36.
    
    # The rank of the all-ones tensor of size a*b*c is... well, it's at most min(a*b, b*c, c*a) = min(6, 18, 12) = 6.
    # Actually, the all-ones tensor can be written as:
    # T(i,j,k) = 1 = (1,1,...,1)·e_i * (1,1,...,1)·e_j * (1,1,...,1)·e_k
    # Wait, that's just one rank-1 term!
    # T(i,j,k) = 1 for all i,j,k, which is the outer product of three all-ones vectors.
    # So the rank is 1!
    
    # But that can't be right for matrix multiplication. Let me reconsider.
    
    # Oh wait, I think I'm confusing two different formulations.
    
    # In the standard matrix multiplication tensor formulation:
    # We have three vector spaces: A (size n*m), B (size m*p), C (size p*n).
    # The tensor T is in A* ⊗ B* ⊗ C* (or A ⊗ B ⊗ C depending on convention).
    # T(a, b, c) = 1 if a, b, c are "compatible", i.e., if a = (i,j), b = (j',k), c = (k',i') and i=i', j=j', k=k'.
    # Otherwise T(a, b, c) = 0.
    
    # So T is not the all-ones tensor. It's a sparse tensor with 36 non-zero entries
    # (one for each valid (i,j,k) triple, where the indices are consistent).
    
    # Specifically:
    # T(i*m+j, j*p+k, k*n+i) = 1 for all i in 0..1, j in 0..2, k in 0..5.
    # All other entries are 0.
    
    # This is the matrix multiplication tensor for (2,3,6).
    
    # Now, what's its rank? It's at least max(6, 18, 12) = 18 (by the "slice rank" bound).
    # And at most 36 (trivial).
    
    # Let me try to find a rank-18 decomposition.
    
    # Idea: For each (j,k) pair (18 pairs), use one term.
    # Term (j,k): U[i*m+j] = 1 for all i, V[j*p+k] = 1, W[k*n+i] = 1 for all i.
    # Contribution to T(a, b, c):
    #   If a = i*m+j, b = j'*p+k', c = k''*n+i', then:
    #   Term (j,k) contributes: U[i*m+j] * V[j*p+k] * W[k*n+i'] = 1 * delta(j,j') * delta(k,k') * 1
    #   = delta(j,j') * delta(k,k')
    #   So it contributes 1 to T(i*m+j, j*p+k, k*n+i') for all i, i'.
    #   But we only want T(i*m+j, j*p+k, k*n+i) = 1 (with i=i'), not for all i'.
    
    # So that doesn't work. We need to separate the i and i' indices.
    
    # Let me try: for each (i,j,k) triple, use one term. That's 36 terms (trivial).
    
    # To reduce, we need to share terms. Note that for fixed (j,k), the terms for i=0 and i=1
    # differ only in the U and W factors.
    
    # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, all others 0.
    # Contribution: 1 to T(i*m+j, j*p+k, k*n+i), 0 elsewhere.
    
    # Can we combine Term (0,j,k) and Term (1,j,k)?
    # Combined term: U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=1.
    # Contribution to T(a, b, c):
    #   If a = i*m+j, b = j'*p+k', c = k''*n+i', then:
    #   Contribution = delta(i,0 or 1) * delta(j,j') * delta(k,k') * delta(i',0 or 1)
    #   = delta(j,j') * delta(k,k') if a has the right j and c has any i'
    #   So it contributes 1 to T(i*m+j, j*p+k, k*n+i') for all i, i'.
    #   But we only want it for i=i'.
    
    # So combining doesn't work directly.
    
    # Let me try a different approach. Use 2 terms per (j,k) pair:
    # Term 1: U[0*m+j]=1, U[1*m+j]=0, V[j*p+k]=1, W[k*n+0]=1, W[k*n+1]=0
    #   -> contributes to T(0*m+j, j*p+k, k*n+0) = T(j, j*p+k, k*n)
    # Term 2: U[0*m+j]=0, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=0, W[k*n+1]=1
    #   -> contributes to T(1*m+j, j*p+k, k*n+1) = T(3+j, j*p+k, k*n+1)
    # Total: 2*18 = 36 terms. Still the same.
    
    # I think for this problem, the rank might actually be 36, or close to it.
    # Let me just return the trivial decomposition for now.
    
    # Actually, wait. Let me reconsider the problem.
    
    # The matrix multiplication tensor for (n,m,p) has a well-known rank.
    # For (2,2,2), the rank is 7 (Strassen).
    # For (2,2,1), the rank is 4.
    # For (2,3,6), I'm not sure what the rank is.
    
    # Let me try to find a lower-rank decomposition using a more systematic approach.
    
    # I'll use the following strategy:
    # 1. Start with the trivial rank-36 decomposition.
    # 2. Try to reduce the rank by combining terms.
    
    # For now, let me just return the trivial decomposition.
    
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
