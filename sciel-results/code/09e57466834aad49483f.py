"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For [2,5,6], use a rank-55 decomposition
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can save 5 multiplications by sharing work between the two rows
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each j in 0..4, k in 0..5: we need A[0,j]*B[j,k] and A[1,j]*B[j,k]
    # Instead of 2*5*6=60 terms, we use 5*6=30 terms for row 0 and 5*6=30 for row 1,
    # but we can merge 5 terms where we compute A[0,j]*B[j,k] + A[1,j]*B[j,k] style combos.
    # Actually, simplest: just do 55 terms by dropping 5 redundant ones via linear dependence.
    
    # Standard construction with 55 terms:
    # For j=0..4, k=0..5: term for (i=0) -> 30 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1  # A[0, j]
            V[j * p + k][col] = 1  # B[j, k]
            W[k * n + 0][col] = 1  # C[0, k]
            col += 1
    
    # For j=0..4, k=0..5: term for (i=1) -> 30 terms, but we'll only do 25
    # to get total 55. Actually, let's just do 30+25=55.
    for j in range(m):
        for k in range(p):
            if j * p + k < 25:  # only 25 of the 30
                U[1 * m + j][col] = 1  # A[1, j]
                V[j * p + k][col] = 1  # B[j, k]
                W[k * n + 1][col] = 1  # C[1, k]
                col += 1
    # This gives 30 + 25 = 55 terms, but it's NOT correct - we'd be missing terms.
    
    # This approach is flawed. Let me use a proper rank-55 construction.
    # The correct way: the tensor [2,5,6] has rank exactly 60 in general, but
    # for specific structured inputs it can be less. However, for exact decomposition
    # of the full tensor, the rank is 60.
    
    # Actually, the rank of [2,m,p] is min(2mp, ...) but for generic m,p it's 2mp.
    # So the schoolbook rank of 60 may be optimal for [2,5,6].
    
    # Let me return the schoolbook decomposition which is known correct.
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
# EVOLVE-BLOCK-END
