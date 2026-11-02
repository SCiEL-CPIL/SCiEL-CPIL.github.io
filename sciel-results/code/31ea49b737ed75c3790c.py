"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        a = r // p
        b = r % p
        
        # U row index is i * m + j. We set it to 1 if i == a, for all j.
        for j in range(m):
            U[a * m + j][r] = 1
            
        # V row index is j * p + k. We set it to 1 for all j, k.
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
                
        # W row index is k * n + i. We set it to 1 if k == b, for all i.
        for i in range(n):
            W[b * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
