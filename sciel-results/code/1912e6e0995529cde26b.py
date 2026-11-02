"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    # a in 0..n-1, b in 0..m-1, l in 0..p/m-1 (assuming p is multiple of m)
    # Here n=2, m=3, p=9. p/m = 3.
    # r corresponds to (a, b, l)
    
    if p % m != 0:
        # Fallback to schoolbook if not divisible, though for this task it is
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    groups = p // m
    rank = n * m * groups
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for a in range(n):
        for b in range(m):
            for l in range(groups):
                # U[i*m+j, col] = 1 if i==a, j==b
                U[a * m + b][column] = 1
                
                # V[j*p+k, col] = 1 if floor(k/m) == l
                # k ranges 0..p-1.
                # l-th group is k in [l*m, (l+1)*m - 1]
                for j in range(m):
                    for k in range(l * m, (l + 1) * m):
                        V[j * p + k][column] = 1
                        
                # W[k*n+i, col] = 1 for all k, i
                for k in range(p):
                    for i in range(n):
                        W[k * n + i][column] = 1
                        
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
