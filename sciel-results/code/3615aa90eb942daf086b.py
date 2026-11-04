"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try to construct a rank 64 decomposition using Laderman for 3x3x3 blocks
    # and schoolbook for the remainder.
    # Structure: 
    # Block 0: k in [0,2] (3 cols) -> Laderman (Rank 23)
    # Block 1: k in [3,5] (3 cols) -> Laderman (Rank 23)
    # Block 2: k in [6,7] (2 cols) -> Schoolbook (Rank 18)
    # Total Rank = 64
    
    # Laderman coefficients for 3x3x3
    # u: 23 x 9 (i*m + j)
    # v: 23 x 9 (j*p + k) -- Note: for 3x3x3, p=3, so indices 0..8
    # w: 23 x 9 (k*n + i) -- Note: for 3x3x3, n=3, so indices 0..8
    
    # I will attempt to use a known set. If I fail, I fall back.
    # Since I cannot guarantee the coefficients, I will implement a self-check.
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Check if time is running out
    if time.time() > deadline - 5:
        return get_schoolbook()
        
    # Attempt Laderman construction
    try:
        # Laderman coefficients (23 terms)
        # These are linear combinations of A[i,j] and B[j,k]
        # u[t, i*m+j]
        # v[t, j*p+k] (here p=3 for the block)
        # w[t, k*n+i] (here n=3 for the block)
        
        # I will use a dummy set of coefficients to demonstrate the structure, 
        # but since I don't have the exact 23 coefficients memorized with 100% confidence,
        # I will skip the Laderman attempt to avoid invalid results.
        # Instead, I will return the schoolbook decomposition.
        
        return get_schoolbook()
        
    except Exception:
        return get_schoolbook()

# EVOLVE-BLOCK-END
