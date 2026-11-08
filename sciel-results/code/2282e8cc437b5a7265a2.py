"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # T[i,j,k,l] = delta_{j,k}, independent of i,l
    # Try rank = m = 4: for each c in 0..3, set U[i*m+c, c]=1, V[c*p+c, c]=1, W[c*n+0, c]=1... 
    # But W index is k*n+i, so W[c*n+i, c] for i in 0,1
    # Let's try: for c in 0..3: U[i*m+c, c] = 1 for all i, V[c*p+c, c] = 1, W[c*n+i, c] = 1 for all i
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c [j==c] [k==c] [1] = delta_{j,k}
    # But this gives T[i,j,k,l] = delta_{j,k} for all i,l, which is what we want!
    rank = m  # = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(m):
        for i in range(n):
            U[i * m + c][c] = 1
            W[c * n + i][c] = 1
        V[c * p + c][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
