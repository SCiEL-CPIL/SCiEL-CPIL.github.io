DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use the schoolbook decomposition but with a simple optimization:
    # For the (4,4,4) corner, use a rank-16 block instead of rank-64
    # by exploiting that r(2,2,2) <= 7 and r(4,4,4) <= r(2,2,2)*r(4,4,4)/r(2,2,2) ...
    # Actually, let's just use rank 1 for each (i,j,k) triple but note that
    # the minimal rank is at most 100 (schoolbook). 
    # Return schoolbook as valid baseline.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
