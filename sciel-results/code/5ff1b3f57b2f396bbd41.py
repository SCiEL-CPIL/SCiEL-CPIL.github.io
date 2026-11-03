"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Structural approach: Use block decomposition
    # C (4x8) = [C_left (4x4) | C_right (4x4)]
    # C_left[i,k] = sum_j A[i,j] * B[j,k] for k in 0..3
    # C_right[i,k] = sum_j A[i,j] * B[j,4+k] for k in 0..3
    
    # Key insight: Both blocks share the same A matrix
    # We can use a decomposition that exploits this shared structure
    
    # For each 4x4x4 block, we'll use a rank-16 decomposition based on
    # the following observation: we can group the 16 terms in each block
    # and use linear combinations to reduce the effective rank.
    
    # Actually, let's use a more systematic approach:
    # The tensor for 4x4x8 has structure that allows us to use
    # a "shared A" decomposition.
    
    # Let's construct a decomposition where:
    # - U has shape (16, R)
    # - V has shape (32, R)  
    # - W has shape (32, R)
    
    # We'll try to find the minimal R by using the algebraic structure.
    
    # For now, let's use a rank-64 decomposition that exploits block structure:
    # Split into 4 blocks of 2x2x4, each using Strassen-like decomposition
    
    # Actually, let's try a direct construction:
    # Use the fact that for each output element C[i,k], we need sum_j A[i,j]*B[j,k]
    # We can group the j dimension.
    
    # Let's try rank 64 first as a safe improvement over 128:
    # Group pairs of j indices and use linear combinations
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 32 x 64
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each (i,j,k) triple gets its own term in the first half
                if col < rank:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # For the remaining terms, we can try to use linear combinations
    # But for now, let's just use the first 64 terms and leave the rest as 0
    # This gives us a valid rank-64 decomposition (not minimal, but better than 128)
    
    # Actually, this won't work because we need all 128 terms to be represented.
    # Let me reconsider.
    
    # The correct approach: we need to represent all 128 tensor components
    # with only 64 rank-1 terms. This requires that some terms are linear
    # combinations of others.
    
    # For matrix multiplication, this is possible due to the bilinear structure.
    # Let's use a known technique: for each fixed i, the terms over j,k
    # can be compressed.
    
    # Actually, let's just return the schoolbook for now but with a different
    # representation that might allow future optimization.
    
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
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
