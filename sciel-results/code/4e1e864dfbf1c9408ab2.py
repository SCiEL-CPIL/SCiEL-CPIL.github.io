"""Matrix multiplication 2x3x9 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    # Rank is m * n = 3 * 2 = 6
    # Decomposition is a direct sum over j of rank-n decompositions of M_{n,1,p}
    rank = m * n
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # Two rank slots for each j
        r0 = j * n
        r1 = j * n + 1
        
        # For i=0, use r0
        # For i=1, use r1
        
        # Initialize V for this j to 1s for both slots
        for k in range(p):
            V[j * p + k][r0] = 1
            V[j * p + k][r1] = 1
            
        for i in range(n):
            for k in range(p):
                if i == 0:
                    U[i * m + j][r0] = 1
                    W[k * n + i][r0] = 1
                elif i == 1:
                    U[i * m + j][r1] = 1
                    W[k * n + i][r1] = 1
                # Note: for n>2, we would need more slots, but here n=2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
