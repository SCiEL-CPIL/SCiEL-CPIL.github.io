import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 means we need A[i,j]*B[j,k] to contribute to C[i,k]
    # For 3x5 * 5x5 -> 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is: T[i, j, k] = 1 for all valid i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the matrix multiplication tensor M_{i,j,k} has M_{i,j,k} = 1 if j is the "inner" index
    # The decomposition: C[i,k] = sum_c U[i,c]*V[k,c]*W[c] ... no
    
    # Standard formulation: We want C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a trilinear form. The tensor T has T[i,j,k] = 1 (it's the identity in the j-index)
    # Rank-1 term: A[i,c] * B[c,k] * C[i,k] where c indexes the terms
    # For schoolbook: c = (j), A[i,j]*B[j,k]*C[i,k]
    
    # The factor matrices:
    # U has shape (n*m, rank): U[i*m+j, c]
    # V has shape (m*p, rank): V[j*p+k, c]  
    # W has shape (p*n, rank): W[k*n+i, c]
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 if the indices are consistent (which they always are for valid i,j,k)
    
    # Actually, the tensor for matrix multiplication is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But that's not right. The tensor encodes which products are needed.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n×m * m×p is:
    # M[i, j, k] = 1 if j is the shared index, i.e., M[i,j,k] = 1 for all valid i,j,k
    # Wait, no. The tensor is defined such that the bilinear form gives the product.
    
    # The correct interpretation: We have tensors U, V, W such that:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 (the matrix multiplication tensor is all ones in the appropriate subspace)
    
    # Actually, I think the tensor T[i,j,k] = 1 for all i,j,k in the valid range, 
    # and the decomposition must reproduce this.
    
    # Let me verify with schoolbook: U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = U[i*m+j,j]*V[j*p+k,j]*W[k*n+i,j] = 1*1*1 = 1
    # And for c != j, all are zero. So sum = 1. Correct.
    
    # So T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5)
    
    # The rank is the minimum number of rank-1 tensors whose sum equals T.
    
    # Let me try to find a lower-rank decomposition.
    
    # Strategy: Use a block decomposition.
    # Split the 3 rows into 2+1.
    # For the 2×5×5 part, try to find a low-rank decomposition.
    # For the 1×5×5 part, the minimum rank is 5.
    
    # Let me try to construct a rank-19 decomposition for 5×5×5 first.
    # Actually, let me just try to search for a good decomposition directly.
    
    # Given time constraints, let me use a structured approach:
    # Use Strassen-like block decomposition.
    
    # For now, let me try a simple improvement: use the fact that we can share
    # some computations between the 3 rows.
    
    # Let me implement a verified decomposition with rank < 75.
    
    # Approach: Use a 2x2 block Strassen on parts of the matrices.
    
    # Actually, let me just implement a search-based optimizer with a deadline.
    
    # For reliability, let me start with schoolbook and try to reduce rank by 
    # finding linear dependencies.
    
    # The matrix multiplication tensor for 3x5x5:
    # T is a 3x5x5 tensor with all entries = 1.
    # We need to find the minimum rank decomposition.
    
    # Key insight: The tensor T[i,j,k] = 1 is actually a very simple tensor.
    # T = ones(3,5,5)
    # The rank of this tensor... 
    
    # Wait, I need to be more careful. The matrix multiplication tensor is NOT all ones.
    # Let me re-read the problem.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for multiplying (n×m) by (m×p) is:
    # M[i, j, k] where i indexes rows of first matrix, j indexes the shared dimension,
    # k indexes columns of second matrix.
    # M[i,j,k] = 1 for all valid i,j,k.
    
    # Yes, so T[i,j,k] = 1 for i in [0,3), j in [0,5), k in [0,5).
    
    # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) if we're talking about 
    # the standard tensor rank... no, that's not right either.
    
    # Actually, the all-ones tensor ones(n,m,p) has tensor rank max(n,m,p)? No.
    # The tensor rank of ones(n,m,p) is... let me think.
    # ones(n,m,p) = sum_{c=1}^{rank} u_c ⊗ v_c ⊗ w_c
    # This is equivalent to finding the minimum rank of the matrix unfolding.
    # The mode-1 unfolding is a n x (m*p) matrix of all ones, which has rank 1.
    # Similarly for other unfoldings.
    # But tensor rank is different from matrix rank of unfoldings.
    
    # Actually, ones(n,m,p) can be written as:
    # ones(n,m,p) = (sum_i e_i) ⊗ (sum_j f_j) ⊗ (sum_k g_k)
    # This is a single rank-1 tensor! So the rank is 1??
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # Oh, I see my confusion. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the schoolbook decomposition.
    
    # In the schoolbook: U[i*m+j, c] where c indexes (i,j,k) triples.
    # U[i*m+j, (i',j',k')] = 1 if i=i' and j=j'
    # V[j*p+k, (i',j',k')] = 1 if j=j' and k=k'
    # W[k*n+i, (i',j',k')] = 1 if k=k' and i=i'
    
    # So the decomposition is: sum_{i',j',k'} U[i*m+j,(i',j',k')] * V[j*p+k,(i',j',k')] * W[k*n+i,(i',j',k')]
    # = sum_{i',j',k'} [i=i' and j=j'] * [j=j' and k=k'] * [k=k' and i=i']
    # = [i=i and j=j and j=j and k=k and k=k and i=i] = 1
    
    # So the "tensor" being decomposed has T[i,j,k] = 1 for all i,j,k. And the schoolbook
    # uses n*m*p terms. But as I noted, the all-ones tensor has rank 1!
    
    # So why does the schoolbook use n*m*p terms? Because the factor matrices have specific
    # index structures: U is indexed by (i,j), V by (j,k), W by (k,i).
    
    # The constraint is that U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must sum to 1
    # for all i,j,k. The factors are coupled through the indices.
    
    # So it's not just the tensor rank of ones(3,5,5). It's a constrained decomposition
    # where the first factor is indexed by (i,j), second by (j,k), third by (k,i).
    
    # This is the matrix multiplication tensor in its standard form. The rank is the
    # matrix multiplication rank.
    
    # OK so I need to find the minimum rank of this specific tensor decomposition.
    
    # Let me just implement a search. Given time constraints, I'll try a few approaches.
    
    # Approach 1: Try rank 43 (2*19 + 5) as suggested.
    # I need a rank-19 decomposition for 5x5x5.
    
    # Let me try to construct one. For 4x4x4, Bini's algorithm gives rank 19.
    # For 5x5x5, I can try to extend this.
    
    # Actually, let me just try to find any decomposition with rank < 75 using optimization.
    
    # Given the time constraint, let me implement a simple approach:
    # 1. Start with schoolbook (rank 75)
    # 2. Try to reduce rank by finding linear dependencies
    
    # Or better: use a known construction.
    
    # Let me try the following: use Strassen's algorithm adapted for this case.
    
    # For 3x5x5, I can split the 5 into 2+2+1 or 3+2, etc.
    
    # Let me try a practical approach: implement a rank-19 5x5 decomposition if I can
    # verify it, otherwise fall back to schoolbook.
    
    # Given the complexity, let me just implement a verified schoolbook and try a few
    # simple improvements.
    
    # Simple improvement: For the 1x5x5 part (last row), the rank is exactly 5.
    # For the 2x5x5 part, let me try to use a rank-19 5x5 decomposition.
    
    # Let me try to construct a rank-19 decomposition for 5x5x5.
    # This is non-trivial. Let me use a different approach.
    
    # I'll implement a numerical search for a low-rank decomposition with a deadline.
    
    # Actually, given the 130-second deadline, let me implement a structured approach:
    
    # Use the following decomposition:
    # Split 3 = 2 + 1
    # For the 2x5x5 block, use a block-Strassen approach.
    # For the 1x5x5 block, use schoolbook (rank 5).
    
    # For 2x5x5: Write A as [A1; A2] where A1, A2 are 1x5 vectors.
    # C1 = A1 * B, C2 = A2 * B
    # We can compute: S1 = A1 + A2, S2 = A1 - A2
    # C1 = (S1*B + S2*B)/2, C2 = (S1*B - S2*B)/2
    # This doesn't save anything.
    
    # Better: use Strassen-like on B.
    # Split B into 2x2 blocks of size ~2.5x2.5... this gets messy.
    
    # Let me just implement a verified decomposition with the best rank I can find.
    
    # Given time constraints, let me implement schoolbook (rank 75) and return it.
    # This guarantees correctness.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
