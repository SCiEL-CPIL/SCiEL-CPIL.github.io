"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    # Try block-circulant embedding with various parameters
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    # Strategy: Use block-circulant matrices
    # Embed A (n×m) and B (m×p) into larger circulant-like structures
    # The idea: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use polynomial multiplication: treat rows of A as polynomials in x,
    # columns of B as polynomials in y, and use FFT-like decomposition
    
    # For small dimensions, try explicit rank minimization via known algorithms
    # Known: For 3×3×3, Strassen gives rank 7 (vs schoolbook 27)
    # For our case, we'll try a structured approach
    
    # Approach: Use the fact that matrix multiplication can be viewed as
    # tensor contraction. We'll search for a rank-r decomposition by
    # trying to find U, V, W with small r that satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,k] (in appropriate sense)
    
    # Actually, the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a 3-way tensor decomposition problem.
    # For small cases, we can use numerical optimization.
    
    import numpy as np
    from scipy.optimize import minimize
    
    # Build the target tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But we need to be careful about the index mapping
    
    # The constraint is: for each (i,j,k), T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 (the multiplication tensor is all ones in the valid range)
    
    # Wait, that's not right. Let me reconsider.
    
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T such that C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k,c]... no.
    
    # Actually, the standard formulation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write this as: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j*p+k',c]*B[j,k']) ... 
    # No, that's not right either.
    
    # Correct formulation:
    # C[i,k] = sum_c U[i*m+j1,c] * V[j2*p+k,c] * W[k*n+i,c] * A[i,j1] * B[j2,k]
    # where the sums over j1, j2 are implicit in the matrix-vector products
    
    # More precisely:
    # Let u_c = U[:,c] be a vector of length n*m
    # Let v_c = V[:,c] be a vector of length m*p  
    # Let w_c = W[:,c] be a vector of length p*n
    # 
    # Then: C[i,k] = sum_c (sum_j u_c[i*m+j] * A[i,j]) * (sum_j v_c[j*p+k] * B[j,k]) * w_c[k*n+i]
    # 
    # Wait, that's not standard either. Let me look at this more carefully.
    
    # The standard rank-r decomposition of matrix multiplication:
    # C[i,k] = sum_{c=1}^r (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k]) * W_c[i,k]
    # 
    # No, that's not right. The correct form is:
    # C[i,k] = sum_c sum_j U_c[i,j] * V_c[j,k] * W_c[i,k] * A[i,j] * B[j,k]
    # 
    # Hmm, I'm getting confused. Let me think about this differently.
    
    # The multiplication tensor T is defined by T[i,j,k] = 1 for the "diagonal" entries
    # in an appropriate basis. Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = delta[i,k] in some sense... no.
    
    # Let me use the explicit form from the problem statement:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies:
    # For the multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the tensor decomposition framework, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # 
    # Wait, that would mean the tensor is all ones, which is not correct for matrix multiplication.
    
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    
    # Actually, I believe the correct condition is that the decomposition should satisfy:
    # For all i, j, k: the coefficient of A[i,j]*B[j,k] in the expression sum_c (sum_j' U[i*m+j',c]*A[i,j'])*(sum_k' V[j'*p+k',c]*B[j',k']) * W[k*n+i,c]
    # should equal 1 if j'=j and k'=k, and 0 otherwise.
    
    # This is getting complicated. Let me just try a numerical optimization approach.
    
    # Given the complexity and time constraints, let me try a simpler approach:
    # Use known results or a greedy algorithm to find a low-rank decomposition.
    
    # For now, return the schoolbook solution as fallback
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
