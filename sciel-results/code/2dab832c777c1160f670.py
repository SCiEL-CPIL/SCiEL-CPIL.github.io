"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The schoolbook algorithm has rank 3*3*7 = 63.
    For small matrix multiplication tensors, finding a rank-minimal decomposition
    significantly lower than the schoolbook bound is a difficult open problem in
    algebraic complexity theory, often requiring complex algebraic identities
    (like Strassen's for 2x2) or heavy computational search.
    
    Given the constraints (120s time limit, need for exact integer/half-integer
    coefficients, and the specific dimensions 3x3x7), the most robust and
    guaranteed exact decomposition is the schoolbook algorithm.
    
    We construct the factor matrices U, V, W such that the tensor T = sum_r U_r \otimes V_r \otimes W_r
    exactly represents the matrix multiplication map.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all indices i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The term A[i,j] * B[j,k] contributes to C[i,k]
                # In the tensor decomposition, this corresponds to a rank-1 term
                # U[i*m + j, column] = 1
                # V[j*p + k, column] = 1
                # W[k*n + i, column] = 1
                # Note: The problem statement specifies W[k*n+i, r] for output index (k,i) 
                # which corresponds to C[i,k] if we map C index (i,k) to flat index k*n+i?
                # Let's check the convention: "output factor follows the official trace convention W[k*n+i,r]"
                # This implies the output matrix C is indexed by (i,k) but stored in W as row k*n+i.
                # So C[i,k] corresponds to W row k*n + i.
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verification is optional but good for safety if time permits.
    # For this simple construction, it is guaranteed correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
