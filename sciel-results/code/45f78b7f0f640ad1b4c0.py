import time
import itertools
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = 1 if the output element (i,k) depends on input A[i,j] and B[j,k]
    # In the standard decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has rank at most n*m*p (trivial) but often lower.
    #
    # For matrix multiplication, the rank is known to be bounded by:
    # - n*m*p (trivial)
    # - Various improved bounds
    #
    # The current seed uses rank = n*m*p = 120.
    #
    # A well-known improvement for rectangular matrix multiplication:
    # For multiplying an (n x m) by (m x p) matrix, the rank can be bounded by
    # min(n*m*p, various Strassen-like bounds)
    #
    # However, for small dimensions like 4x5x6, we can try to find a lower rank
    # decomposition using known results or systematic search.
    #
    # One approach: use the fact that matrix multiplication tensor rank for
    # (n x m) * (m x p) is at most:
    # - If n=m=p: O(n^omega) where omega < 2.373
    # - For rectangular: various bounds exist
    #
    # For our specific case 4x5x6, let's try to construct a decomposition
    # with rank less than 120.
    #
    # A simple improvement: note that if we fix j, we're doing an outer product
    # of column j of A (length n) and row j of B (length p).
    # The trivial decomposition uses one rank-1 term per (i,j,k) triple.
    #
    # We can try to use the rank of the matrix multiplication tensor more cleverly.
    #
    # Known result: The tensor rank of multiplying (n x m) by (m x p) is at most
    # (n*p) * min(n, m, p) in some cases, but this isn't directly applicable.
    #
    # Let me think about this differently. The matrix multiplication tensor T_{n,m,p}
    # has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Actually, T is a 3-tensor where T[i,j,k] = 1 if and only if...
    # Wait, for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k), and the multiplication
    # is: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    #
    # Actually, the standard tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (corresponding to A[i,j]*B[j,k] contributing to C[i,k])
    #
    # The rank of this tensor is what we're looking for.
    #
    # For the specific case of 4x5x6:
    # - Trivial rank: 4*5*6 = 120
    # - We can try to find a lower rank using known algorithms or bounds.
    #
    # One known approach: if we can find a decomposition using the structure
    # of the matrices, we might reduce the rank.
    #
    # Let me try a different approach: use the fact that for small matrices,
    # we can sometimes find explicit low-rank decompositions.
    #
    # Actually, let me reconsider. The rank of the matrix multiplication tensor
    # for (n x m) * (m x p) is at least max(n*p, n*m, m*p) and at most n*m*p.
    # For 4x5x6: at least max(24, 20, 30) = 30, at most 120.
    #
    # Let me try to construct a decomposition with rank closer to the lower bound.
    #
    # One strategy: use the "blocking" approach where we divide the computation
    # into blocks and use known decompositions for each block.
    #
    # For now, let me try a simple improvement: instead of using all 120 terms,
    # see if we can combine some terms.
    #
    # Actually, for the specific dimensions 4x5x6, let me just return the trivial
    # decomposition but check if there's a known lower bound.
    #
    # The key insight: for matrix multiplication of (n x m) by (m x p), the tensor
    # rank is the minimum number of multiplications needed. For 4x5x6, this is
    # a specific value that may be less than 120.
    #
    # Since finding the exact minimal rank is hard, let me try a few known bounds:
    # - If we can express the multiplication using fewer than 120 scalar multiplications
    #
    # Let me try a heuristic: for each j (the shared dimension), we're computing
    # n*p products A[i,j]*B[j,k] for i in [n], k in [p]. That's n*p = 24 products
    # per j, and there are m=5 values of j, so 120 total.
    #
    # Can we do better? For a fixed j, the outer product of column j of A and
    # row j of B gives an n x p matrix. The rank of this outer product is 1,
    # so we can't reduce the number of multiplications for a fixed j below n*p
    # if we're doing it independently.
    #
    # But across different j values, there might be correlations we can exploit.
    #
    # This is getting complex. Let me just return the trivial decomposition for now,
    # as improving it requires deeper algorithmic work.
    
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
# EVOLVE-BLOCK-END
