"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 16 terms for j=2 part
    # 28 terms for j=0,1 part (4 blocks of Strassen rank 7)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=2 (A_{i,2} * B_{2,k})
    # A is 2x3, B is 3x8.
    # U index: i*m + j. Here j=2.
    # V index: j*p + k. Here j=2.
    # W index: k*n + i.
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=0,1 using Strassen on 4 blocks of B
    # A_1 is A[:, 0:2] (2x2)
    # B_1 is B[0:2, :] (2x8)
    # Split B_1 into 4 blocks of 2x2: cols (0,1), (2,3), (4,5), (6,7)
    for l in range(4):
        k0 = 4 * l
        k1 = 4 * l + 1
        
        # M1: (a+d)(e+h)
        # U: a+d -> U[0,0]=1, U[1,0]=1, U[0,1]=1, U[1,1]=1
        # V: e+h -> V[0,k0]=1, V[1,k1]=1
        # W: P00 (C[0,k0]), P11 (C[1,k1])
        U[0 * m + 0][col] = 1
        U[1 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2: (c+d)e
        # U: c+d -> U[1,0]=1, U[1,1]=1
        # V: e -> V[0,k0]=1
        # W: P10 (C[1,k0]), P11 (C[1,k1]) neg
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3: a(f-h)
        # U: a -> U[0,0]=1
        # V: f-h -> V[0,k1]=1, V[1,k1]=-1
        # W: P01 (C[0,k1]), P11 (C[1,k1])
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4: d(g-e)
        # U: d -> U[1,1]=1
        # V: g-e -> V[1,k0]=1, V[0,k0]=-1
        # W: P00 (C[0,k0]), P10 (C[1,k0])
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5: (a+b)h
        # U: a+b -> U[0,0]=1, U[0,1]=1
        # V: h -> V[1,k1]=1
        # W: P00 (C[0,k0]) neg, P01 (C[0,k1])
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6: (c-a)(e+f)
        # U: c-a -> U[1,0]=1, U[0,0]=-1
        # V: e+f -> V[0,k0]=1, V[0,k1]=1
        # W: P11 (C[1,k1])
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7: (b-d)(g+h)
        # U: b-d -> U[0,1]=1, U[1,1]=-1
        # V: g+h -> V[1,k0]=1, V[1,k1]=1
        # W: P00 (C[0,k0])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
