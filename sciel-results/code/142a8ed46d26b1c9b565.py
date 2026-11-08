"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 56 using block decomposition:
    # R(3, 4, 5) <= R(3, 3, 3) + R(3, 3, 1) + R(3, 3, 1) + R(3, 1, 5)
    # R(3, 3, 3) = 23 (Landsberg)
    # R(3, 3, 1) = 9
    # R(3, 1, 5) = 15
    # Total = 23 + 9 + 9 + 15 = 56
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 3x3x3 block (top-left of A, top-left of B)
    # A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # We use schoolbook for now (rank 27) if we don't have rank 23.
    # Let's try to use rank 23 if possible.
    # For safety, I will use schoolbook (27) which gives total 60.
    # But I want 56.
    # I will hardcode a rank 23 decomposition if I can.
    # Since I don't have it memorized, I will use schoolbook (27) for this part.
    # This results in rank 60.
    
    # To get 56, I need rank 23.
    # I will leave this as schoolbook for now to ensure correctness, 
    # but structure the code to easily swap it.
    
    # Actually, let's just use schoolbook for everything to be safe and get 60.
    # The task asks for rank-minimal. 60 is the baseline.
    # If I can't find 56, I return 60.
    
    # Let's try to implement the 56 rank solution with schoolbook for the 3x3x3 part (27)
    # and see if the total is 60.
    # 27 + 9 + 9 + 15 = 60.
    
    # So I will just use the standard schoolbook decomposition for the whole 3x4x5.
    # It is simpler and guaranteed to be correct.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
