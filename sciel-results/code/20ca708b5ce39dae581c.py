"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1

    # Try to construct a rank 27 decomposition using Kronecker product
    # of M_{2,3,3} (rank 9) and M_{1,1,3} (rank 3)
    # This requires finding a rank 9 decomposition for 2x3x3
    
    # Known decomposition for 2x3x3 with rank 9?
    # Let's try to use a simple heuristic or a known set of coefficients.
    # If we fail, we return schoolbook.
    
    # For now, we return schoolbook as it is guaranteed valid.
    # Future evolutions might find the rank 27 decomposition.
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
