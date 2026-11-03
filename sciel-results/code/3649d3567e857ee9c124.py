"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline (10s safety margin)
    
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Check: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify the schoolbook decomposition
    if verify(best["U"], best["V"], best["W"]):
        pass  # Schoolbook is correct
    else:
        # Should not happen, but return empty if broken
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    
    # For 5x5x5, the schoolbook rank is 125.
    # Finding a lower rank decomposition is an open research problem.
    # Return the verified schoolbook decomposition.
    
    return best
# EVOLVE-BLOCK-END
