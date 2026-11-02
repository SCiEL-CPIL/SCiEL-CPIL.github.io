import time
import sys

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x9 matrix.
    
    Currently implements the schoolbook algorithm with rank 54.
    This is a valid baseline.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
                # Time check
                if time.time() > deadline:
                    # Return best found so far (which is the full schoolbook if we finish, 
                    # but if we timeout mid-loop, we return incomplete? 
                    # The task says "return the best result found". 
                    # An incomplete decomposition is invalid. 
                    # So we should not check time inside the loop for a deterministic small task.
                    # 54 iterations is fast.
                    pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
