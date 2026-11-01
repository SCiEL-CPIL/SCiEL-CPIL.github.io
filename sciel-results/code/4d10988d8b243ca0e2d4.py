"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Modular decomposition strategy:
    # Split the multiplication M_{2,3,5} into M_{2,3,3} and M_{2,3,2} based on the k-index.
    # The W-dimension supports are disjoint, so ranks are additive.
    # Currently using schoolbook for both sub-problems, resulting in rank 30.
    # This structure allows for future insertion of optimized sub-decompositions (e.g., rank 14 for 2x3x3).
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: k in [0, 2] (size 3)
    p1 = 3
    # Sub-problem 2: k in [3, 4] (size 2)
    p2 = 2
    
    rank1 = n * m * p1
    rank2 = n * m * p2
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Sub-problem 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Fill Sub-problem 2
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
