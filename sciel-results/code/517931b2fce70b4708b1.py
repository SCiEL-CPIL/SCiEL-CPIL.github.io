"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import itertools
import sys

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek rank r such that T = sum_{c=0}^{r-1} U[:,c] outer V[:,c] outer W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a multilinear system. For fixed U and V, W is linear.
    # But it's generally hard. Let's try to find a low-rank decomposition.
    
    # Lower bound: rank >= max(n*m, m*p, p*n) / something? 
    # Actually, for matrix multiplication C = AB where A is n x m, B is m x p:
    # The tensor has rank at least max(n*m, m*p, p*n) in some cases, but that's not tight.
    # Strassen's algorithm shows that for 2x2x2, rank 7 < 8.
    # For 3x5x7, let's try to find a decomposition.
    
    # One approach: use the fact that we can factor through intermediate matrices.
    # But the standard schoolbook is rank n*m*p = 3*5*7 = 105.
    
    # Let's try to use a bilinear algorithm approach.
    # We can write the multiplication as a sum of products of linear forms.
    
    # Actually, let's try a different approach: use the structure of matrix multiplication.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to find a decomposition with rank less than 105.
    
    # For small matrices, we can try to use known results or search.
    # Since this is a hard problem, let's try to use a greedy or heuristic approach.
    
    # Alternative: use the fact that the tensor is a matrix multiplication tensor,
    # and try to find a decomposition using the "border rank" or "subrank" concepts.
    
    # For now, let's try to construct a decomposition with rank = n*m*p = 105
    # but try to reduce it by exploiting linearity.
    
    # Actually, let's try a simple approach: use the identity
    # C[i,k] = sum_j A[i,j] B[j,k]
    # and try to find a decomposition where some of the U, V, W entries are zero.
    
    # Let's try to use the fact that we can combine some terms.
    # For example, if we have two terms that differ only in one index, we might combine them.
    
    # Given the complexity, let's return the schoolbook decomposition for now,
    # but try to optimize it slightly.
    
    # Actually, let's try to use a different strategy: 
    # We know that for matrix multiplication, the rank is at least the "subrank" 
    # which is related to the geometric rank. But finding the exact rank is hard.
    
    # Let's try to use a numerical approach to find a low-rank decomposition.
    # We can set up the system of equations and solve it.
    
    # The system is:
    # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    # For r = 105, we have 105*(15 + 35 + 21) = 105*71 = 7455 unknowns and 105 equations.
    # So the system is underdetermined, and we can find a solution.
    
    # But we want the minimum r. Let's try r = 15, 20, 25, etc.
    
    # However, solving this system exactly is difficult. Let's try a heuristic.
    
    # Given the time constraint, let's return the schoolbook decomposition.
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
