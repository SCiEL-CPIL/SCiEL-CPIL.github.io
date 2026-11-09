"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Try to find a lower rank decomposition using Strassen-like block decomposition
    # For 3x5x6, we can try to partition the matrices and use block multiplication
    
    # Strategy: Try to find a decomposition with rank < 90
    # We'll use a systematic search over small rank values with integer/half-integer entries
    
    # First, let's try a structured approach based on Strassen's algorithm for 2x2 blocks
    # Partition A (3x5) into blocks, B (5x6) into blocks
    
    # Since dimensions are small, let's try to find exact decompositions with rank < 90
    # by solving the tensor decomposition problem directly
    
    # The tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if and only if we're multiplying A[i,j] * B[j,k]
    # We want to find rank-1 tensors that sum to T
    
    # For a rank-r decomposition, we need:
    # T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a decomposition with rank slightly less than 90
    # by exploiting the structure of the problem
    
    # Approach: Use the fact that for small matrices, we can try to find
    # decompositions with integer entries by solving a system of equations
    
    # For now, let's return the schoolbook decomposition as a valid baseline
    # A more sophisticated search would be needed to find lower rank decompositions
    
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
