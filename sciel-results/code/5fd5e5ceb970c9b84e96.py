def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Rank 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x10 schoolbook (Rank 20)
    # A[0..1, 0] * B[0, 0..9]
    # U index: i*m + 0 = i*3
    # V index: 0*p + k = k
    # W index: k*n + i = k*2 + i
    r = 0
    for i in range(2):
        for k in range(10):
            U[i * 3 + 0][r] = 1
            V[0 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
            
    # Part 2: 2x2x10 Strassen (Rank 35)
    # A[0..1, 1..2] * B[1..2, 0..9]
    # Split B into 5 blocks of 2 columns.
    # Block b covers cols 2b, 2b+1.
    # A indices:
    # a = U[0*3+1] = U[1]
    # b = U[0*3+2] = U[2]
    # c = U[1*3+1] = U[4]
    # d = U[1*3+2] = U[5]
    
    # B indices for block b:
    # e = V[1*10 + 2b] = V[10+2b]
    # f = V[1*10 + 2b+1] = V[11+2b]
    # g = V[2*10 + 2b] = V[20+2b]
    # h = V[2*10 + 2b+1] = V[21+2b]
    
    # W indices for block b:
    # C00 = W[2b*2 + 0] = W[4b]
    # C01 = W[(2b+1)*2 + 0] = W[4b+2]
    # C10 = W[2b*2 + 1] = W[4b+1]
    # C11 = W[(2b+1)*2 + 1] = W[4b+3]
    
    # Strassen Factors
    # 1: (a+d)(e+h) -> W00:1, W11:1
    # 2: (c+d)e     -> W10:1, W11:-1
    # 3: a(f-h)     -> W01:1, W11:1
    # 4: d(g-e)     -> W00:-1, W01:1
    # 5: (a+b)h     -> W00:1
    # 6: (c-d)(f+g) -> W11:1
    # 7: (b-c)(e+g) -> W10:1, W00:-1
    
    for b in range(5):
        # Base rank for this block
        rb = r
        
        # Factor 1
        U[1][rb+0] = 1
        U[5][rb+0] = 1
        V[10 + 2*b][rb+0] = 1
        V[21 + 2*b][rb+0] = 1
        W[4*b][rb+0] = 1
        W[4*b+3][rb+0] = 1
        
        # Factor 2
        U[4][rb+1] = 1
        U[5][rb+1] = 1
        V[10 + 2*b][rb+1] = 1
        W[4*b+1][rb+1] = 1
        W[4*b+3][rb+1] = -1
        
        # Factor 3
        U[1][rb+2] = 1
        V[11 + 2*b][rb+2] = 1
        V[21 + 2*b][rb+2] = -1
        W[4*b+2][rb+2] = 1
        W[4*b+3][rb+2] = 1
        
        # Factor 4
        U[5][rb+3] = 1
        V[20 + 2*b][rb+3] = 1
        V[10 + 2*b][rb+3] = -1
        W[4*b][rb+3] = -1
        W[4*b+2][rb+3] = 1
        
        # Factor 5
        U[1][rb+4] = 1
        U[2][rb+4] = 1
        V[21 + 2*b][rb+4] = 1
        W[4*b][rb+4] = 1
        
        # Factor 6
        U[4][rb+5] = 1
        U[5][rb+5] = -1
        V[11 + 2*b][rb+5] = 1
        V[20 + 2*b][rb+5] = 1
        W[4*b+3][rb+5] = 1
        
        # Factor 7
        U[2][rb+6] = 1
        U[4][rb+6] = -1
        V[10 + 2*b][rb+6] = 1
        V[20 + 2*b][rb+6] = 1
        W[4*b+1][rb+6] = 1
        W[4*b][rb+6] = -1
        
        r += 7
        
    return {"U": U, "V": V, "W": W}
