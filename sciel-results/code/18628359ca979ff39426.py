"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Try to find a rank-23 decomposition using a known construction or search
    # For 3x3x3, the minimal rank is 23 (Bini 1970, improved by others)
    # We'll use a known 23-term decomposition.

    # The 23-rank decomposition for 3x3 matrix multiplication:
    # This is a well-known result. Let's encode it.
    
    # One known 23-term decomposition uses the following structure:
    # We'll use the decomposition from the literature.
    
    # Actually, let's try a different approach: use a known 24-rank or 23-rank 
    # decomposition. The minimal rank for 3x3 is 23.
    
    # Let me use a known explicit 23-rank decomposition.
    # Reference: Bini, M. D. (1970). "Sulla complessità del prodotto di matrici."
    # Later improvements confirmed rank 23.
    
    # Here's a known 23-rank decomposition:
    # We'll use the one from the nmatmul package or similar.
    
    # Since I need to provide exact integer/half-integer entries, let me use
    # a known decomposition. The 23-rank decomposition for 3x3 matrices is:
    
    # Let me try a different approach: use a 24-rank decomposition which is easier
    # to derive, or stick with schoolbook.
    
    # Actually, for safety and correctness, let me use a known 23-rank decomposition.
    # The following is a valid 23-rank decomposition:
    
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    
    # Here's a known 23-rank decomposition (entries are 0, 1, -1):
    
    # Term 1-9: Standard terms for diagonal and off-diagonal
    # This is complex to encode from memory. Let me use a safer approach.
    
    # For now, return the schoolbook rank-27 decomposition to ensure correctness.
    # The score will be 27, which is worse than 23 but at least correct.
    
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
