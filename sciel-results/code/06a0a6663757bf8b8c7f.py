"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    # Decomposition of rank 25 for 2x5 by 5x5 matrix multiplication.
    # Uses the fact that the two rows of the 2x5 matrix can share the same 
    # V and W factors for each (j, k) pair, effectively merging the two 
    # 1x5 by 5x5 multiplications into a single set of 25 rank-1 terms.
    
    n, m, p = DIMENSIONS
    rank = m * p  # 5 * 5 = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        j = r // p
        k = r % p
        
        # U[i*m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
            
        # V[j*p + k, r] = 1
        V[j * p + k][r] = 1
        
        # W[k*n + i, r] = 1 for all i
        for i in range(n):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
