"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Uses schoolbook decomposition (rank 144) with a search for potential improvements.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Schoolbook decomposition: rank = n*m*p = 144
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Attempt to find a lower-rank decomposition using iterative methods
    # The tensor T[i,j,k] = 1 for all valid indices in the multiplication
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try rank reduction via randomized SVD-like approach
    # This is a research-level problem; for now, return schoolbook
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        """Convert numpy array to list, rounding to nearest half-integer."""
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
