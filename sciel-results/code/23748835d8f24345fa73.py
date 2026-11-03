"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of size 2x3x10.
    Rank is n*m*p = 60.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin of 10s before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time periodically (though loop is very fast)
                if time.time() > deadline:
                    # If we hit the deadline (unlikely for this simple loop), 
                    # we return what we have. Since it's schoolbook, we should 
                    # finish instantly. This is a safety guard.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
