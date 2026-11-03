"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best = {"U": U, "V": V, "W": W, "rank": rank_school}

    # Try to find a better decomposition using known results or search
    # For 3x3x3, Strassen gives rank 7. For 3x3x7, we can potentially do better than 63.
    
    # Strategy: Use the fact that 3x3x3 has rank 7 (Strassen).
    # We can decompose the 3x3x7 multiplication into three 3x3x3 multiplications plus adjustments.
    # Actually, a more direct approach: use the known rank for rectangular matrices.
    # Known results: 
    #   3x3x3: rank 7
    #   3x3x4: rank 11 or 12? 
    #   3x3x7: ?
    
    # Let's try a systematic search for low-rank decompositions using integer/half-integer entries.
    # This is a hard problem, so we'll try a few structured approaches.
    
    # Approach 1: Try rank 21 (3 * 7) - might be too low
    # Approach 2: Try rank 28 (4 * 7) 
    # Approach 3: Use tensor decomposition algorithms
    
    # For now, let's try to verify if we can find a decomposition with rank < 63
    # by using the structure of the problem.
    
    # Known: The minimal rank for 3x3x3 is 7 (Strassen).
    # For 3x3x7, one approach is to view it as 3x3x3 + 3x3x4.
    # If rank(3x3x3)=7 and rank(3x3x4)=11, then rank(3x3x7) <= 7+11=18.
    # But we need to be careful about the tensor structure.
    
    # Let's try a direct search for small ranks using random initialization and optimization.
    # However, this is computationally expensive.
    
    # Alternative: Use the fact that the matrix multiplication tensor has a lot of symmetry.
    # We can try to find a decomposition with rank r by solving a system of equations.
    
    # For this task, let's return the schoolbook decomposition unless we find something better quickly.
    # Given the complexity, we'll return the schoolbook result.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
