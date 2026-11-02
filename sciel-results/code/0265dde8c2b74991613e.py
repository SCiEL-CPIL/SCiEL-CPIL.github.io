import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: k=0,1 (2x3x2) Rank 8
    # Block 2: k=2,3 (2x3x2) Rank 8
    # Block 3: k=4 (2x3x1) Rank 6
    
    # Helper for 2x3x2 Rank 8
    def add_2x3x2_rank8(col, k_off):
        # This is a placeholder for the actual rank 8 algorithm
        # I will implement a specific one here.
        # ...
        return col

    # Helper for 2x3x1 Rank 6
    def add_2x3x1_rank6(col, k_off):
        for i in range(2):
            for j in range(3):
                U[i*m + j, col] = 1
                V[j*p + k_off, col] = 1
                W[k_off*n + i, col] = 1
                col += 1
        return col

    # Implement Rank 8 for 2x3x2
    # Using a known decomposition
    # ...
    
    # Fallback if rank 8 is too hard to code blindly:
    # Use 28 rank solution.
    
    pass
