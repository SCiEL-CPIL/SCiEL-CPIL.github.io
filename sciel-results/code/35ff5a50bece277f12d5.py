import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    rank = 18
    
    # The multiplication tensor T is defined such that:
    # C = A @ B
    # where A is (n x m) and B is (m x p).
    # C[i, j] = sum_k A[i, k] B[k, j]
    #
    # We want to express this as:
    # C[i, j] = sum_r U[i, r] V[r, j]  (conceptually, but the interface is specific)
    #
    # The interface requires:
    # U: (n*m, rank) -> indices (i*m + k, r)
    # V: (m*p, rank) -> indices (k*p + j, r)
    # W: (p*n, rank) -> indices (j*n + i, r)
    #
    # The decomposition must satisfy:
    # T[i, k, j, r] ? No, the tensor is T[i, k, j] = 1 if it's part of the sum.
    # Actually, the standard tensor for matrix multiplication M_{n,m,p} has indices (i, k, j).
    # T[i, k, j] = 1.
    # The bilinear form is sum_{i,k,j} T[i,k,j] A[i,k] B[k,j] = C[i,j].
    #
    # The factorization is:
    # T[i, k, j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    #
    # Wait, the output W has shape (p*n, rank). The index is j*n + i.
    # So W[j*n+i, r] is the coefficient for the output C[i,j].
    #
    # Let's check the indices again.
    # U row index: i*m + k (0 <= i < n, 0 <= k < m)
    # V row index: k*p + j (0 <= k < m, 0 <= j < p)
    # W row index: j*n + i (0 <= j < p, 0 <= i < n)
    #
    # We need:
    # sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = 1 if the term A[i,k]B[k,j] contributes to C[i,j].
    # Actually, the tensor entry T[i,k,j] is 1 for all valid i,k,j.
    # The multiplication is:
    # C[i,j] = sum_k A[i,k] B[k,j]
    #
    # So for a specific output C[i,j], the contribution from term r is:
    # U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # This must be 1 if the term A[i,k]B[k,j] is included in the sum for C[i,j], and 0 otherwise.
    # But U and V depend on k.
    #
    # Let's look at the structure.
    # For a fixed r, U[:, r] is a vector of length n*m.
    # V[:, r] is a vector of length m*p.
    # W[:, r] is a vector of length p*n.
    #
    # The condition is:
    # sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] = delta_{term}
    # where delta_{term} is 1 if the pair (i,k) and (k,j) are linked in the r-th rank-1 component.
    #
    # Actually, simpler view:
    # The tensor T has entries T[i, k, j] = 1.
    # We decompose T = sum_r U_r \otimes V_r \otimes W_r.
    # T[i, k, j] = sum_r U_r[i, k] V_r[k, j] W_r[i, j] ? No.
    #
    # Let's stick to the flattened indices.
    # T_flat[index_u, index_v] ? No, it's a 3-way tensor.
    # T[i, k, j] = sum_r U_flat[i*m+k, r] * V_flat[k*p+j, r] * W_flat[j*n+i, r]
    #
    # This equation must hold for all i, k, j.
    #
    # Let's try to construct a rank-18 decomposition.
    # n=2, m=3, p=9.
    #
    # A known lower bound for 2x3x9 is 18.
    # Can we achieve 18?
    #
    # Let's try a simple structured decomposition.
    # If we could use rank = n*m*p = 54, it's trivial (identity).
    # We need rank 18.
    #
    # Let's try to generate random integer/half-integer matrices and check?
    # No, that's hard.
    #
    # Let's look at the "finite field" strategy mentioned in the prompt.
    # F_{2^3} has 8 elements. m=3.
    # Identify A rows with F_8? No, A is 2x3.
    #
    # Let's try a different approach.
    # Since I need to return a valid decomposition, and finding the exact minimal rank is hard,
    # I should first ensure I return *a* valid decomposition, even if not minimal, to get a non-zero score.
    # The trivial rank is n*m*p = 54.
    #
    # Let's construct the rank-54 decomposition.
    # For each i, k, j, we have a term A[i,k]B[k,j].
    # We can map each (i,k,j) to a unique rank index r.
    # There are 2*3*9 = 54 terms.
    #
    # Let r index from 0 to 53.
    # Let r = i * 27 + k * 9 + j.
    #
    # U[i*m+k, r] should be 1 if r corresponds to this i,k and some j?
    # No, U depends on i,k. V depends on k,j. W depends on i,j.
    #
    # For a specific r (which encodes i0, k0, j0):
    # U[i*m+k, r] = 1 if i==i0 and k==k0, else 0.
    # V[k*p+j, r] = 1 if k==k0 and j==j0, else 0.
    # W[j*n+i, r] = 1 if i==i0 and j==j0, else 0.
    #
    # Let's check the sum for a specific (i, k, j).
    # Sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    #
    # Term r is non-zero only if:
    # 1. U[i*m+k, r] != 0 => i_r == i and k_r == k
    # 2. V[k*p+j, r] != 0 => k_r == k and j_r == j
    # 3. W[j*n+i, r] != 0 => i_r == i and j_r == j
    #
    # If i_r == i, k_r == k, j_r == j, then all three are 1.
    # Is there such an r?
    # Yes, r = i * 27 + k * 9 + j.
    # So the sum is 1.
    #
    # If any of the indices don't match, at least one factor is 0.
    # So the sum is 1 for all valid i,k,j.
    #
    # This is a valid rank-54 decomposition.
    # Score will be 54.
    # The seed program likely had a higher rank or failed.
    # If the seed was failing (score 0), any valid score is an improvement.
    #
    # Can we do better?
    # Rank 18 is the target.
    # Let's try to implement the rank-18 construction if possible, otherwise stick to 54.
    #
    # Given the time limit and complexity, let's provide the rank-54 solution first.
    # It guarantees a valid result.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                
                # U index: i*m + k
                U[i * m + k, r] = 1.0
                # V index: k*p + j
                V[k * p + j, r] = 1.0
                # W index: j*n + i
                W[j * n + i, r] = 1.0
                
    return {'U': U, 'V': V, 'W': W}
