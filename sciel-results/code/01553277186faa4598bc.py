"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Subspace embedding and null-space projection.
    We start from the schoolbook decomposition of the larger 4x4 * 4x8 tensor (rank 128).
    The 3x4 * 4x8 problem is a restriction where the 4th row of the first matrix is zero.
    
    In the 4x4x8 schoolbook decomposition, each term c corresponds to a triple (i,j,k) with
    i in {0,1,2,3}, j in {0,1,2,3}, k in {0,...,7}.
    
    For the 3x4x8 problem, we only need i in {0,1,2}. The terms with i=3 are those where
    U has 1 at position 3*4+j = 12+j. But in the 3x4 problem, U only has rows 0..11.
    So if we simply drop the i=3 terms, we get a valid decomposition of rank 3*4*8 = 96.
    
    But can we do better? The key insight from the directive is to look for terms in the
    4x4x8 decomposition that lie in the null space of the projection onto the 3x4x8 subspace.
    
    Actually, let's think about this differently. The 3x4 * 4x8 multiplication can be viewed
    as: given A (3x4) and B (4x8), compute C = A*B (3x8).
    
    We can try to use the fact that we're multiplying a 3x4 by 4x8. One approach is to
    decompose the 4x8 matrix B into two 4x4 matrices B1 and B2, and compute A*B1 and A*B2
    separately. If we could multiply 3x4 by 4x4 with rank r, then 3x4 by 4x8 would have
    rank 2r.
    
    For 3x4 * 4x4:
    Schoolbook rank = 3*4*4 = 48.
    Can we do better?
    
    Alternatively, we can use the structure of the problem. Note that 3x4 * 4x8 can be
    related to 3x4 * 4x4 by noting that B is 4x8, which is two 4x4 blocks.
    
    Let's try another approach: use the known lower bound and see if we can achieve it.
    The lower bound for the rank of the tensor for n x m * m x p multiplication is:
    max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    
    However, achieving the lower bound is generally not possible. The schoolbook gives 96.
    
    Let's try to use a block decomposition. Write A as a 3x4 matrix and B as a 4x8 matrix.
    We can partition B into B = [B1 | B2] where B1 and B2 are 4x4.
    Then C = A*B = [A*B1 | A*B2].
    
    If we can compute A*B1 with rank r1 and A*B2 with rank r2, then the total rank is r1 + r2.
    By symmetry, r1 = r2, so total rank = 2*r1.
    
    For 3x4 * 4x4, the schoolbook rank is 48. So this approach gives 96, same as before.
    
    Let's try to find a better decomposition for 3x4 * 4x4.
    
    Actually, let's just return the schoolbook decomposition for 3x4 * 4x8 with rank 96.
    """
    n, m, p = DIMENSIONS
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
