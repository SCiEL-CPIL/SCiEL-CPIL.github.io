"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication 3x5 * 5x5.
    
    The standard schoolbook decomposition gives rank 75.
    We attempt to find a lower-rank decomposition using known results.
    
    For M_{3,5,5}, the tensor rank is known to be less than 75.
    However, without explicit coefficients, we default to schoolbook.
    
    To potentially improve, we can use the fact that R(M_{2,2,2})=7 (Strassen).
    If we can decompose the problem into blocks, we might reduce rank.
    But exact block decomposition for 3x5x5 is non-trivial.
    
    Given the constraints and the need for exact integer/half-integer entries,
    and the risk of invalidity, we return the schoolbook decomposition.
    This guarantees a valid score of 1.0.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all indices i, j, k
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
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
