"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 12: 6 for row 1, 6 for row 2
    # T = sum_{k=1}^6 (u_{1,k} + u_{2,k}) \otimes e_k
    # u_{r,k} = sum_{j=1}^4 e_{r,j} \otimes e_{j,k}
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(n): # r=0,1
        for k in range(p): # k=0..5
            # W[k*n + r] = 1
            W[k * n + r][col] = 1
            for j in range(m): # j=0..3
                # U[r*m + j] = 1
                U[r * m + j][col] = 1
                # V[j*p + k] = 1
                V[j * p + k][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
