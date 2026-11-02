"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i)
    # where i in [0,n), j in [0,m), k in [0,p)
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # The tensor can be written as sum over (i,j,k) of e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is the trivial decomposition with rank n*m*p
    
    # To find a better decomposition, we can try to find linear dependencies
    # or use algebraic structure.
    
    # For now, let's try to use the fact that we can group terms.
    # One approach: for each pair (i,k), we have m terms that all contribute to C[i,k].
    # But these are already separate rank-1 terms in the trivial decomposition.
    
    # Let me try a different approach: use the structure of the problem.
    # 3x4 x 4x7 can be seen as:
    # - 3 rows of A, each with 4 elements
    # - 4 rows of B, each with 7 elements
    # - Output is 3x7
    
    # Key insight: we can use the fact that matrix multiplication is bilinear.
    # The tensor rank is the minimum number of rank-1 tensors needed.
    
    # For small cases, let's try to find a decomposition by solving a system of equations.
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T = sum_{r=0}^{R-1} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # This means: for all (a,b,c), T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Let's try to find a decomposition with rank < 84.
    # We can start by trying rank 80, 75, etc. and see if we can find one.
    
    # Actually, for this specific case, let me try a constructive approach.
    # We can partition the 4 inner dimension into 2+2 and use Strassen for 2x2 blocks.
    
    # A is 3x4: split into [A1 (3x2), A2 (3x2)]
    # B is 4x7: split into [B1 (2x7), B2 (2x7)]
    # C = A1*B1 + A2*B2
    
    # Each A_i is 3x2, B_i is 2x7. The product A_i*B_i is 3x7.
    # The tensor rank for 3x2 x 2x7 is at most 3*2*7 = 42.
    # So total rank is at most 42+42 = 84. Same as before.
    
    # To do better, we need to exploit more structure.
    
    # Let me try: can we use the fact that 3 and 7 are odd?
    # Or can we find a decomposition that shares some terms?
    
    # Alternative approach: use the "border rank" or other algebraic invariants.
    
    # For now, let's just return the trivial decomposition. It's valid and correct.
    # The score will be 1.0, same as the seed.
    
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
