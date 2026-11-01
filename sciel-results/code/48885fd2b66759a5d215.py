def construct():
    n, m, p = 2, 2, 12
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 4 blocks
    for t in range(4):
        base = t * 9
        # V indices
        e = 3*t
        f = 3*t + 1
        g = 3*t + 2
        h = 12 + 3*t
        i = 12 + 3*t + 1
        j = 12 + 3*t + 2
        
        # W indices
        W00 = 6*t
        W10 = 6*t + 1
        W01 = 6*t + 2
        W11 = 6*t + 3
        W02 = 6*t + 4
        W12 = 6*t + 5
        
        # Strassen 7 products
        # M1=(a+d)(e+h)
        U[0][base+0] = 1; U[3][base+0] = 1
        V[e][base+0] = 1; V[h][base+0] = 1
        W[W00][base+0] = 1; W[W11][base+0] = 1
        
        # M2=(c+d)e
        U[2][base+1] = 1; U[3][base+1] = 1
        V[e][base+1] = 1
        W[W10][base+1] = 1; W[W11][base+1] = -1
        
        # M3=a(f-h)
        U[0][base+2] = 1
        V[f][base+2] = 1; V[h][base+2] = -1
        W[W01][base+2] = 1; W[W11][base+2] = 1
        
        # M4=d(i-h)
        U[3][base+3] = 1
        V[i][base+3] = 1; V[h][base+3] = -1
        W[W00][base+3] = 1; W[W10][base+3] = 1
        
        # M5=(a+b)i
        U[0][base+4] = 1; U[1][base+4] = 1
        V[i][base+4] = 1
        W[W00][base+4] = -1; W[W01][base+4] = 1
        
        # M6=(c-a)(e+f)
        U[2][base+5] = 1; U[0][base+5] = -1
        V[e][base+5] = 1; V[f][base+5] = 1
        W[W11][base+5] = 1
        
        # M7=(b-d)(h+i)
        U[1][base+6] = 1; U[3][base+6] = -1
        V[h][base+6] = 1; V[i][base+6] = 1
        W[W00][base+6] = 1
        
        # Extra 2 products for col 2
        # P8=(a-c)g
        U[0][base+7] = 1; U[2][base+7] = -1
        V[g][base+7] = 1
        W[W02][base+7] = 1; W[W12][base+7] = -1
        
        # P9=(b-d)j
        U[1][base+8] = 1; U[3][base+8] = -1
        V[j][base+8] = 1
        W[W02][base+8] = 1; W[W12][base+8] = -1

    # Check and return
