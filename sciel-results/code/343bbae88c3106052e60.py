"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 64
    
    # The schoolbook decomposition
    def schoolbook():
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
    
    # Try to find a better decomposition using Strassen-like ideas
    # For 2x2 * 2x16, we can view it as 4 blocks of 2x2 * 2x4
    # Or we can try to use the structure more directly
    
    # Let's try a systematic search for rank < 64
    # The tensor has 4 * 2 * 16 = 128 nonzero entries in its flat representation
    # Each rank-1 term can cover at most 4 entries (one from each matrix dimension)
    # So theoretically rank >= 32, but likely higher
    
    # Let's try to construct a decomposition using the fact that 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed (i,j), we have C[i,k] += A[i,j] * B[j,k] for all k
    
    # A natural approach: use the 4 values of A as "switches"
    # But that gives rank 4*16 = 64 (schoolbook)
    
    # Alternative: use polynomial/FFT structure
    # If we view the 16 columns as a polynomial of degree 15 in B's second index,
    # then C[i,x] = sum_j A[i,j] * B[j,x] is a polynomial identity
    
    # Let's try rank 32 first with a structured approach
    # Use the fact that we can combine columns
    
    # Actually, let's just try to verify that schoolbook works and return it
    # A more sophisticated approach would be needed to beat 64
    
    return schoolbook()
# EVOLVE-BLOCK-END
