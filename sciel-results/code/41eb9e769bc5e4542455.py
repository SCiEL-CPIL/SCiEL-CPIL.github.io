"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We need to find a rank-minimal exact tensor decomposition for
    # multiplying a 3x3 matrix by a 3x5 matrix.
    
    # The schoolbook algorithm has rank n*m*p = 45.
    # We need to find if we can do better.
    
    # Let's think about this more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is 3x3, B is 3x5, C is 3x5.
    
    # The tensor has 3*3*5 = 45 terms in the naive decomposition.
    
    # Can we use any structure to reduce the rank?
    # For standard matrix multiplication m x n @ n x p, the rank is at most m*n*p.
    # Known lower bounds exist, but for small sizes like 3x3 @ 3x5, 
    # the exact minimum rank may not be well-known or may equal 45.
    
    # Let's try to implement a search for a lower-rank decomposition.
    # We'll use a greedy or iterative approach to see if we can find rank < 45.
    
    # Actually, let's first verify that the schoolbook rank-45 works,
    # then try to find improvements.
    
    # For now, let's implement a more systematic approach.
    # We'll try to find a decomposition with rank < 45 by solving a system of equations.
    
    # The matrix multiplication tensor T is defined by:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, that's not quite right. Let me be more careful.
    
    # The tensor for matrix multiplication (3x3) @ (3x5) -> (3x5) is:
    # T_{i,j,k} where i in [0,3), j in [0,3), k in [0,5)
    # T_{i,j,k} = 1 (this is the standard Kronecker product structure)
    
    # Actually, the tensor is:
    # For each output element C[i,k], we have:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # So the tensor T has components T[i,j,k] = 1 for all i,j,k,
    # and the multiplication is: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    # To decompose this tensor, we want to write:
    # T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, I need to be more careful with the indexing.
    # U has shape (n*m, R) = (9, R)
    # V has shape (m*p, R) = (15, R)  
    # W has shape (p*n, R) = (15, R)
    
    # The decomposition condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a system of equations. For R=45, we have the trivial solution
    # where each (i,j,k) triple gets its own rank-1 term.
    
    # Can we do better? Let's try R=44 and see if a solution exists.
    # This would require solving a nonlinear system, which is hard.
    
    # Alternative approach: use known results or heuristics.
    
    # For now, let's just return the schoolbook rank-45 decomposition
    # to ensure correctness, and then we can try to optimize.
    
    rank = n * m * p  # 45
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
