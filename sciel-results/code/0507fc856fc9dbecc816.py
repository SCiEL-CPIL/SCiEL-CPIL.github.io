"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Time guard: Although this is O(1) for fixed dims, we respect the directive.
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill
    
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed program style, but could use numpy for speed if rank was large.
    # Here rank is 60, so lists are fine and satisfy "integer/half-integer entries" (integers are fine).
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    # Return whatever we have so far? 
                    # For schoolbook, we must complete it to be valid.
                    # If we abort, it's invalid. 
                    # Since it's fast, this won't trigger.
                    pass
                
                # Index calculations based on task contract
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
