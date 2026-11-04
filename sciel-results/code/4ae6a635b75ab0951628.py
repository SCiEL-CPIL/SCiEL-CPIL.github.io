"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Try to find a lower-rank decomposition
    # For M_{2,5,5}, the tensor is T[i,j,k] = delta[i,k'] * delta[j,k''] type structure
    # Actually T[i,j,k] is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # We want to decompose this as sum of rank-1 tensors
    
    # Schoolbook gives rank 50. Let's see if we can do better.
    # For 2x5x5, try rank 45, 40, etc.
    
    # Use a heuristic: try random initializations and gradient descent
    # But first, let's check if there's a known result
    
    # Actually, let me try a structured approach.
    # For each k in 0..4, we compute two outputs: C[0,k] and C[1,k]
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # This is 10 dot products of length 5, total 50 mults.
    # No obvious sharing since A[0,j] and A[1,j] are independent.
    
    # Let's just use schoolbook for now, as it's guaranteed correct
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
