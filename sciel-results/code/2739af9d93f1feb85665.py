"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For matrices of size (n x m) and (m x p), the rank is n * m * p.
    
    Includes an explicit deadline guard to ensure completion within the 
    evaluator's 150s limit, returning the best valid result found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # In the unlikely event of timeout during construction (highly improbable for schoolbook),
                    # we return the partial result. However, schoolbook is O(1) construction time.
                    return {"U": U, "V": V, "W": W}
                
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
