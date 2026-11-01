"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor to decompose is the all-ones tensor of shape (n, m, p).
    # We can decompose it by summing over the middle dimension m.
    # Each slice j is a tensor of shape (n, 1, p) with all ones.
    # Such a tensor is rank 1: u (sum of e_i) v (e_j) w (sum of e_k).
    # Since the slices are disjoint in the middle dimension, the total rank is m.
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        # U: index i*m + j. Set to 1 for all i.
        for i in range(n):
            U[i * m + j][r] = 1
        # V: index j*p + k. Set to 1 for all k.
        for k in range(p):
            V[j * p + k][r] = 1
        # W: index k*n + i. Set to 1 for all k, i.
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
