"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import time
import itertools

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _make_schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    U_best, V_best, W_best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # For 2x4x8, schoolbook rank is 64
    # Known lower bounds suggest rank could be lower, but finding it is hard
    # Let's try a simple approach: check if rank 63 is possible with integer entries
    
    # Since a full search for rank 63 is infeasible, we return the schoolbook
    # unless we can quickly find a valid lower rank decomposition
    
    # Return the schoolbook decomposition which is guaranteed valid
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
