"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T for 3x3 * 3x7 multiplication has indices (i,j,k) where
    # i in 0..2 (row of A), j in 0..2 (col of A / row of B), k in 0..6 (col of B)
    # T[i,j,k] = 1 for all valid indices
    # We need to find a rank decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector of length n*m=9, V_r is length m*p=21, W_r is length p*n=21
    
    # Strategy: Use the block structure. The 3x7 matrix B can be seen as [B1 | B2 | B3]
    # where B1, B2 are 3x3 and B3 is 3x1.
    # C = A * B = A*B1 + A*B2 + A*B3
    
    # For the two 3x3 blocks, we can use Strassen's algorithm for 2x2 sub-blocks
    # and handle the remaining parts separately.
    
    # Actually, let's try a simpler approach first: use the known rank-23 decomposition
    # for 3x3 matrix multiplication (if it exists) or use Strassen-based decomposition.
    
    # Strassen for 2x2 uses 7 multiplications. For 3x3, we can pad to 4x4 and use Strassen
    # recursively, but that gives rank 7^2 = 49 for 4x4, which is worse than schoolbook 27.
    
    # Better: decompose 3x3 into 2x2 + 1x1 parts.
    # A = [[A11, a12, a13],
    #      [a21, A22, a23],
    #      [a31, a32, a33]]
    # where A11 is 2x2, and the rest are scalars/vectors.
    
    # Actually, let me think about this differently. The schoolbook rank is 63.
    # We know that for 3x3 * 3x3, the minimum rank is at most 23 (Landsberg's bound)
    # but the best known exact decomposition has rank 23 or less.
    
    # Let's try to use a known rank-23 decomposition for 3x3 * 3x3 if available,
    # and handle the 3x1 part with rank 9.
    
    # For now, let's use a randomized search to find a lower-rank decomposition.
    # We'll start with the schoolbook decomposition and try to find linear dependencies
    # or use optimization to reduce the rank.
    
    # Actually, let's use a different approach: construct the full tensor and use
    # tensor rank algorithms to find a low-rank decomposition.
    
    # The tensor has size 9 x 21 x 21 = 3969 elements.
    # We can try to find a rank-r decomposition by solving a system of equations.
    
    # For each rank r, we need to find U (9xr), V (21xr), W (21xr) such that
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a nonlinear system. Let's try a simple approach:
    # 1. Start with r = 63 (schoolbook)
    # 2. Try to reduce r by finding linear combinations that cancel out
    
    # Actually, let's just use the schoolbook for now and see if we can find
    # a better decomposition through optimization.
    
    # For a more practical approach, let's use the fact that the 3x1 block
    # can be computed with rank 9, and try to optimize the two 3x3 blocks.
    
    # Let's try rank 50 as a starting point and see if we can find a valid decomposition.
    
    # Actually, I'll implement a randomized search for a lower-rank decomposition.
    # We'll use the schoolbook as a baseline and try to find linear dependencies.
    
    rank = 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now let's try to reduce the rank by finding linear combinations.
    # We'll check if any column is a linear combination of others.
    
    # Actually, for the 3x3 * 3x3 blocks, we can use Strassen's algorithm.
    # Strassen for 2x2 matrices:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For 3x3, we can split into 2x2 and 1x1 parts and use a similar approach.
    
    # Let's implement a 3x3 decomposition based on Strassen-like ideas.
    
    # A = [[a, b, c],
    #      [d, e, f],
    #      [g, h, i]]
    # B = [[j, k, l],
    #      [m, n, o],
    #      [p, q, r]]
    
    # C[0,0] = aj + bm + cp
    # C[0,1] = ak + bn + cq
    # C[0,2] = al + bo + cr
    # C[1,0] = dj + em + fp
    # C[1,1] = dk + en + fq
    # C[1,2] = dl + eo + fr
    # C[2,0] = gj + hm + ip
    # C[2,1] = gk + hn + iq
    # C[2,2] = gl + ho + ir
    
    # We can try to find 7 or fewer products that generate all 9 entries.
    
    # This is complex. For now, let's return the schoolbook decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
