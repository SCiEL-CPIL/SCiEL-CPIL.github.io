def construct():
    n, m, p = 2, 3, 9
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(3):
        # Block l: columns 3l, 3l+1, 3l+2 of B and C
        # Strassen for A[:, 0:2] * B[0:2, 3l:3l+2]
        # M1 = (A00 + A11) * (B0,3l + B1,3l+1)
        U[0][col] = 1; U[3][col] = 1
        V[0*9 + 3*l][col] = 1; V[1*9 + 3*l + 1][col] = 1
        W[3*l*2 + 0][col] = 1; W[3*l*2 + 1][col] = 1; W[(3*l+1)*2 + 1][col] = 1
        # This is getting complex to map Strassen directly.
        # Let's use the explicit terms.
