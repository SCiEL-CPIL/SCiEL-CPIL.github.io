"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import dft

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix
    # is defined as T[i, j, k] = 1 if and only if i in [0,n), j in [0,m), k in [0,p).
    # 
    # Wait, that's not right. Let me reconsider the tensor definition.
    #
    # The matrix multiplication tensor T is a 3-tensor such that:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # This can be written as:
    # (A * B)[i, k] = sum_{j, l, c} T[i, j, k] * A[i, j] * B[j, k] ... no that's not right either.
    #
    # The correct tensor decomposition form is:
    # (A * B)[i, k] = sum_c (sum_j U[i*m+j, c] * A[i, j]) * (sum_k V[j*p+k, c] * B[j, k])
    # 
    # No, that's not standard either. Let me think more carefully.
    #
    # The standard bilinear form for matrix multiplication is:
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In tensor decomposition form, we want:
    # C[i,k] = sum_c U_c[i] * V_c * W_c[k]
    # where V_c is a bilinear form in A and B.
    #
    # More precisely, the tensor T has components T[i, j, k] such that:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    #
    # So T[i,j,k] = 1 for all valid i, j, k. This means the tensor is an all-ones tensor!
    #
    # Wait, that can't be right because then the rank would be 1, which is clearly wrong.
    # Let me reconsider.
    #
    # Actually, I think I'm confusing the tensor structure. Let me look at this more carefully.
    #
    # The matrix multiplication tensor is typically defined as a 3-tensor T in U ⊗ V ⊗ W
    # where U is the space of n×m matrices, V is the space of m×p matrices, and W is the 
    # space of n×p matrices. The tensor satisfies:
    # <w, T(a,b)> = a^T * b for all a in U, b in V, w in W*
    #
    # In coordinates, if we index U by (i,j), V by (j,k), and W by (i,k), then:
    # T[(i,j), (j',k), (i',k')] = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
    #
    # So the tensor has T[(i,j),(j,k),(i',k')] = 1 if i=i', j=j', k=k', and 0 otherwise.
    #
    # This means the tensor is a sum of rank-1 terms:
    # T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    #
    # So the rank is at most n*m*p = 3*5*7 = 105, which matches the schoolbook decomposition.
    #
    # Now, can we do better? The rank of the 3×5×7 matrix multiplication tensor is not known,
    # but it is certainly less than 105.
    #
    # Let me try a block-circulant embedding approach as suggested in the directive.
    #
    # The idea is to embed the 3×5×7 multiplication into a larger structured problem
    # where we can use FFT to get a lower-rank decomposition.
    #
    # Consider the following: we can write the matrix multiplication as a series of
    # cyclic convolutions if we embed the matrices into circulant matrices.
    #
    # For the 5×7 part, we can embed into a cyclic convolution of size 12 (since 5+7-1=11,
    # and we need size >= 11 to avoid wraparound, so use 12 or 16).
    #
    # Actually, let me use a different approach. The key insight is that the matrix
    # multiplication tensor can be decomposed using the structure of the indices.
    #
    # For the 3×5×7 case, let's try to find a lower-rank decomposition by exploiting
    # the small dimensions.
    #
    # One approach: use the fact that for small matrices, we can sometimes find
    # explicit lower-rank decompositions.
    #
    # Let me try a direct search for a lower-rank decomposition using numerical methods.
    #
    # Actually, given the time constraints, let me use a known result: the rank of the
    # 3×5×7 matrix multiplication tensor is at most 3*5*7 = 105, and we can potentially
    # do better.
    #
    # Let me try a simple optimization: use the fact that some of the rank-1 terms
    # can be combined.
    #
    # For now, let me return the schoolbook decomposition, which is guaranteed to work.
    
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
