import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate through each rank component
    for r in range(rank):
        # Map r to i, j, k
        # r = i * m * p + j * p + k
        k = r % p
        r1 = r // p
        j = r1 % m
        i = r1 // m
        
        # Set the corresponding entries to 1
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
