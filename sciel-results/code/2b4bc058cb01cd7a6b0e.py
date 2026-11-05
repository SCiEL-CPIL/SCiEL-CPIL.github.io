import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses the schoolbook decomposition which is guaranteed to be valid.
    Includes a deadline guard to prevent timeout.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, as per seed program
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return what we have so far? 
                    # No, partial decomposition is invalid.
                    # We must return a valid one.
                    # Since construction is O(1) effectively, we won't hit this.
                    pass
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
