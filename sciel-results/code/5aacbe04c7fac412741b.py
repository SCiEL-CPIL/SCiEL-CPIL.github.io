"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the standard schoolbook decomposition for 3x3 matrix multiplication.
    Rank is 27.
    """
    start_time = time.time()
    deadline = 120.0 # Seconds
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for easy conversion to numpy or direct return
    # The interface expects {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # Entries should be integer/half-integer.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (as per task description "trace convention W[k*n+i,r]")
                W[k * n + i][column] = 1
                column += 1
                
                # Check time, though this loop is very fast
                if time.time() > deadline - 5.0:
                    # If we are close to the limit, break and return what we have (which is complete)
                    # Since the loop is deterministic and small, this is unlikely to trigger
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
