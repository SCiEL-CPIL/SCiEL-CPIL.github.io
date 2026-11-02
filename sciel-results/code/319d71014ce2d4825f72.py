"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers)."""
    r = len(U[0])
    # Multiply by 2 to work with integers
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n*m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m*p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                if s != 8:  # 2*2*2=8
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a lower-rank decomposition
    # Start with rank 24 and go down
    best_rank = n * m * p  # 36 schoolbook
    best_result = None
    
    for target_rank in [24, 22, 20, 19, 18]:
        if time.time() - start_time > 100:
            break
        # Try structured search for this rank
        # Use a simple approach: try all combinations of small values
        # This is only feasible for very small ranks, so we use it as a fallback check
        
        # For now, just verify schoolbook as fallback
        pass
    
    # Fallback: schoolbook decomposition
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
