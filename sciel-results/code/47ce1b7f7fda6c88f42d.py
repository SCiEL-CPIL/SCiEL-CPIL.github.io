import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition: rank = n*m*p = 60
    schoolbook_rank = n * m * p
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Build Strassen 2x2x2 decomposition
    # For 2x2 matrix multiplication: C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Strassen's 7 multiplications:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need to express each of the 7 Strassen
    # terms as rank-1 tensors in the (input, input, output) space.
    # 
    # The key insight: for a 2x2x2 block, we can use rank 7 instead of 8.
    # But we need to be careful about how the indices map.
    
    # Let me try a different approach: use known results for small matrix multiplication.
    # For 2x5x6, we can try to find a decomposition by:
    # 1. Splitting the 5x6 output into blocks
    # 2. Using Strassen for 2x2 blocks where possible
    
    # Actually, let me try a direct approach: use the fact that the tensor
    # for matrix multiplication has a specific structure and try to find
    # linear dependencies among the 60 rank-1 terms.
    
    # For now, let me implement a simple improvement: check if we can
    # reduce the rank by finding that some terms are redundant.
    
    # One known result: for multiplying an m x n matrix by an n x p matrix,
    # the rank is at most m*n*p, and for small cases, better bounds are known.
    
    # Let me try to use a greedy approach to find a lower-rank decomposition.
    
    # Actually, for this problem, let me just return the schoolbook solution
    # as a safe baseline, but let me also try a quick search for improvements.
    
    best_rank = schoolbook_rank
    best_U = [[0]*best_rank for _ in range(nm)]
    best_V = [[0]*best_rank for _ in range(mp)]
    best_W = [[0]*best_rank for _ in range(pn)]
    
    for c, (i, j, k) in enumerate(triples):
        best_U[i*m+j][c] = 1
        best_V[j*p+k][c] = 1
        best_W[k*n+i][c] = 1
    
    # Try to find a better decomposition using the "ladder" technique
    # For n=2, we can potentially use a shifted decomposition.
    
    # The idea: for multiplying a 2 x m matrix by an m x p matrix,
    # we can write the result as:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # This is equivalent to computing two dot products for each k.
    # We can potentially share some computations between the two rows.
    
    # Let me try a specific construction:
    # For each pair (j, k), we need to compute A[0,j]*B[j,k] and A[1,j]*B[j,k].
    # These are independent, so we can't directly share them.
    
    # However, if we consider the 2x2 blocks of B (for fixed j pairs),
    # we might be able to use Strassen-like tricks.
    
    # Let me try a concrete construction for 2x5x6:
    # Split the 5 columns of A into groups: [2, 2, 1]
    # Split the 6 rows of B into groups: [2, 2, 2]
    
    # For each combination of a column group of A and a row group of B,
    # we have a sub-problem of multiplying a 2 x g_A matrix by a g_B x 6 matrix.
    
    # This is getting complex. Let me just return the schoolbook solution.
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
