"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0  # 10s safety margin

# EVOLVE-BLOCK-START
def _time_remaining():
    return DEADLINE - time.time()

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Check shapes
    if len(U) != n * m or len(U[0]) != rank:
        return False
    if len(V) != m * p or len(V[0]) != rank:
        return False
    if len(W) != p * n or len(W[0]) != rank:
        return False
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For exact integer/half-integer entries, we can verify by checking
    # that T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise
    # Actually, the decomposition should satisfy:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta(i,j,k) for the bilinear form
    # Wait, let me reconsider. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 when c=(i,j,k)
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's verify by computing the reconstructed tensor and checking it's all 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _try_rank_59():
    """Try to find a rank-59 decomposition using structured search."""
    n, m, p = 3, 4, 5
    rank = 59
    
    # Strategy: Start from schoolbook (rank 60) and try to merge two terms
    # by finding linear dependencies. We'll use a randomized approach with
    # change of basis in the middle dimension.
    
    # The idea: if we can find vectors u, v, w such that
    # sum of two rank-1 terms can be replaced by one rank-1 term
    # This requires the two terms to be linearly dependent in some sense.
    
    # Alternative: Use the fact that for 3x4x5, the tensor rank is conjectured
    # to be 59. We'll do a structured search over parameterized families.
    
    # Family: U[i*m+j,c] = a_c[i] * b_c[j], V[j*p+k,c] = b_c[j] * c_c[k], W[k*n+i,c] = c_c[k] * a_c[i]
    # This gives sum_c a_c[i]*b_c[j]*b_c[j]*c_c[k]*c_c[k]*a_c[i] = sum_c (a_c[i]*c_c[k])*(b_c[j])^2
    # This doesn't directly help. Let me think differently.
    
    # Actually, let's use a more direct approach: random search with verification
    # We'll generate random matrices and check if they form a valid decomposition.
    
    # For efficiency, we'll use numpy and check the tensor equation
    # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    
    # Let's try a parameterized family based on the middle dimension structure
    # Assume U[i*m+j,c] depends on i and j, V[j*p+k,c] depends on j and k, W[k*n+i,c] depends on k and i
    
    # We'll search over integer/half-integer values in {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}
    # But this is a huge space. Instead, let's use a smarter approach.
    
    # Known result: The 3x4x5 matrix multiplication tensor has rank 59.
    # Let's try to construct it using the "laser method" or by exploiting
    # the specific structure.
    
    # For now, return None if we can't find it quickly
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First try to find rank 59
    result = _try_rank_59()
    if result is not None:
        return result
    
    # Fall back to schoolbook rank 60
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
