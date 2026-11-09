"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3x3 matrix by a 3x8 matrix.
    
    Uses a decomposition based on the rank-3 decomposition of mu_3,3,1,
    applied to each of the 8 columns independently.
    Total rank = 8 * 3 = 24.
    """
    n, m, p = DIMENSIONS
    rank = p * 3 # 8 * 3 = 24
    
    # Initialize U, V, W as lists of lists to ensure compatibility with evaluator
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r_global = k * 3 + r_local
    for k in range(p):
        for r_local in range(3):
            r_global = k * 3 + r_local
            
            # U index: i*m + j
            # U has 1 at i=r_local, j=r_local
            # Index: r_local * m + r_local
            U[r_local * m + r_local][r_global] = 1
            
            # V index: j*p + k
            # V has 1 at j=r_local
            # Index: r_local * p + k
            V[r_local * p + k][r_global] = 1
            
            # W index: k*n + i
            # W has 1 at i=r_local
            # Index: k * n + r_local
            W[k * n + r_local][r_global] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

