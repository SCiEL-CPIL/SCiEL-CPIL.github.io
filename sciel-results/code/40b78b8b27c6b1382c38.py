"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses the standard schoolbook algorithm with rank 125.
This is the known minimal rank for the simple schoolbook approach, 
and a guaranteed valid decomposition.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the 5x5x5 matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' mapping to lists of lists (matrices).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Pre-allocate matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # If we somehow timeout (shouldn't happen for 125 terms), 
                    # return current state.
                    return {"U": U, "V": V, "W": W}
                
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][c] = 1
                V[idx_v][c] = 1
                W[idx_w][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
