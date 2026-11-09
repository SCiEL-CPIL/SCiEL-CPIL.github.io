import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns the schoolbook tensor decomposition for 3x3x7 matrix multiplication.
    Rank is 3*3*7 = 63.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, or numpy if allowed/needed for speed
    # The task allows NumPy, but lists are fine for small sizes.
    # However, the output format is {'U': ..., 'V': ..., 'W': ...}
    # The evaluator likely expects arrays or lists. The seed uses lists.
    
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
                
                # Deadline check
                if time.time() > deadline:
                    # Return best effort (which is the current partial or full if fast enough)
                    # Since it's very fast, this shouldn't trigger.
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
