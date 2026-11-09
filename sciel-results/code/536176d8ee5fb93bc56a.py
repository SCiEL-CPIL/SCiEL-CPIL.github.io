"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # We decompose the 3x3x7 multiplication into:
    # 1. 3x3x3 block (k=0,1,2)
    # 2. 3x3x3 block (k=3,4,5)
    # 3. 3x3x1 block (k=6)
    
    # Each 3x3x3 block uses the schoolbook decomposition (rank 9).
    # The 3x3x1 block uses the schoolbook decomposition (rank 9).
    # Total rank = 9 + 9 + 9 = 27.
    
    rank_total = 27
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a schoolbook decomposition for a range of k
    def add_schoolbook_block(k_start, k_end, offset):
        # k ranges from k_start to k_end (exclusive)
        # i, j range 0..n-1, 0..m-1
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        col = offset
        for i in range(n):
            for j in range(m):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    # Block 1: k=0,1,2
    next_col = add_schoolbook_block(0, 3, 0)
    
    # Block 2: k=3,4,5
    next_col = add_schoolbook_block(3, 6, next_col)
    
    # Block 3: k=6
    next_col = add_schoolbook_block(6, 7, next_col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
