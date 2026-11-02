"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is the all-ones tensor.
    # The minimal rank for the all-ones tensor is 1.
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
