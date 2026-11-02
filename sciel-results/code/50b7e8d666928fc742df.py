"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
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
    
    # Try to find a lower-rank exact decomposition
    # The tensor T[i,j,k] = delta_{ik} for matrix multiplication
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = delta_{ik}
    
    # For small matrices, we can try various rank values and check if an exact
    # decomposition exists with integer/half-integer entries.
    
    # The schoolbook rank is 105. The Strassen-like methods can reduce this.
    # For 3x5x7, the theoretical lower bound is at least max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    
    # Let's try to use a structured approach:
    # The multiplication tensor for (n,m,p) has a known structure.
    # For general matrices, the rank is at most n*m*p (schoolbook).
    # 
    # One approach: use the fact that matrix multiplication can be decomposed
    # using block methods or algebraic identities.
    
    # For now, let's try a few specific structured decompositions.
    # 
    # Key insight: For matrix multiplication A (n x m) * B (m x p),
    # we can write C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if the (i,k)-th entry of the product depends on A[i,j] and B[j,k]
    # 
    # More precisely, the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{ik} ... no, that's not right either.
    # 
    # The correct formulation: The multiplication tensor is a 3-tensor T of size 
    # (n*m) x (m*p) x (n*p) where:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and 0 otherwise.
    #
    # Wait, let me re-read the problem statement. The output factor follows 
    # W[k*n+i, r]. So the third index is k*n+i.
    #
    # The tensor T has T[a, b, c] where:
    # a = i*m + j (row of U, corresponding to A[i,j])
    # b = j*p + k (row of V, corresponding to B[j,k])  
    # c = k*n + i (row of W, corresponding to C[i,k])
    #
    # And T[a, b, c] = 1 iff the indices are consistent, i.e.,
    # a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    
    # Let's verify the schoolbook is correct and try to find improvements.
    
    # For a 3x5x7 matrix multiplication, let's see if we can use
    # a lower rank decomposition.
    
    # One known result: for matrix multiplication of n x m by m x p,
    # the rank is at most n*m*p but can be lower using algebraic methods.
    
    # For small cases, we can try to find the exact minimum rank.
    # But this is computationally expensive in general.
    
    # Let's try a specific structured approach:
    # We can try to factor the multiplication using properties of the indices.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The rank is n*m*p = 105.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
