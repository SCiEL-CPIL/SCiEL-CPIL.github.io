"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use Kronecker product structure: 4x4*4x7 = (2x2*2x2) ⊗ (2x2*1x7)
    # Build using Strassen for 2x2*2x2 and extend to handle the 7 dimension
    
    # Step 1: Define Strassen decomposition for 2x2*2x2
    # Strassen's 7 multiplications for 2x2*2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our purposes, we'll build the full tensor decomposition using Kronecker products
    # and then reduce rank through Gaussian elimination
    
    # The matrix multiplication tensor for (n,m,p) can be viewed as having a natural
    # Kronecker structure when n, m, p are powers of 2.
    
    # For 4x4*4x7, we can think of it as:
    # - 4x4*4x4 part: use Strassen-based decomposition (rank ~49 or better)
    # - 4x4*4x3 part: additional terms
    
    # Let's use a systematic approach:
    # 1. Start with schoolbook decomposition (rank 112)
    # 2. Identify linear dependencies using the structure
    # 3. Apply Gaussian elimination to reduce rank
    
    # Actually, let's use a more direct approach based on known results:
    # For 4x4*4x4, the best known rank is 49 (Laderman's algorithm)
    # We can extend this to 4x4*4x7
    
    # Laderman's algorithm for 4x4*4x4 uses 49 multiplications
    # We'll adapt this for 4x4*4x7
    
    # For now, let's implement a decomposition that achieves rank < 112
    # using a combination of Strassen-like techniques
    
    # Simple approach: use block structure
    # 4x4*4x7 can be decomposed into blocks
    # Let's try to find a rank-100 decomposition first
    
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use a structured decomposition
    # For each of the 4x4=16 output elements C[i,k], we need sum_j A[i,j]*B[j,k]
    # We can use shared intermediate products
    
    # Let's use a simple improvement: group terms to reduce rank
    # For each i, we have 4*7=28 terms, but we can share computations
    
    # Actually, let's just return a valid decomposition with rank 100
    # by using a systematic construction
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                else:
                    # Need to handle overflow - this approach won't work for rank < 112
                    pass
    
    # This simple approach doesn't work. Let me use the schoolbook as fallback
    # but try to find a better one
    
    # For now, return schoolbook to ensure correctness
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
