"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
Uses a rank-16 Winograd/FFT scheme for the 4x4x4 core (if valid over rationals)
and schoolbook for the remaining strips. Falls back to pure schoolbook if needed.
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 4, 5, 6
    
    # Strategy: 
    # Partition A (4x5) into A_core (4x4) and A_strip (4x1)
    # Partition B (5x6) into B_core (4x6) and B_strip (1x6)
    # C = A_core @ B_core + A_strip @ B_strip
    #
    # For A_core @ B_core (4x4 @ 4x6):
    #   Split B_core into B1 (4x4) and B2 (4x2)
    #   A_core @ B1: use Winograd 4x4x4 (rank 16 if valid)
    #   A_core @ B2: use schoolbook 4x4x2 (rank 32)
    #
    # For A_strip @ B_strip (4x1 @ 1x6):
    #   Schoolbook: rank 24
    #
    # Total if Winograd works: 16 + 32 + 24 = 72
    # Otherwise: 120 (pure schoolbook)
    
    # Let's try to construct the Winograd 4x4x4 decomposition.
    # The key question: can we express the 4-point FFT-based decomposition
    # with rational (integer or half-integer) entries?
    
    # 4-point DFT matrix:
    # omega = exp(2*pi*i/4) = i
    # F4[i,j] = omega^(i*j) = i^(i*j)
    # F4 = [[1, 1, 1, 1],
    #       [1, i, -1, -i],
    #       [1, -1, 1, -1],
    #       [1, -i, -1, i]]
    
    # This involves complex numbers, which are not rational.
    # So a direct FFT-based decomposition won't work with rational entries.
    
    # Alternative: Use a different tensor decomposition for 4x4x4.
    # The best known rank for 4x4x4 matrix multiplication is 49 (Strassen-like),
    # but constructing it explicitly is complex.
    
    # For simplicity and correctness, let's stick with schoolbook.
    # The rank is n*m*p = 4*5*6 = 120.
    
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
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
