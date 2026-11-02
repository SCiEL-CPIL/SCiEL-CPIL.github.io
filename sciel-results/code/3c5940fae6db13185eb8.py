"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication 2x3 * 3x6.
    Rank is n*m*p = 36.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0.0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time just in case, though this is O(1) effectively
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                if column >= rank:
                    break
            if column >= rank:
                break
        if column >= rank:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
