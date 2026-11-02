def construct():
    n, m, p = 2, 2, 14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(7):
        for t in range(7):
            col = k * 7 + t
            # U indices: 0:a, 1:c, 2:e, 3:f
            # V indices: 2k:b00, 2k+1:b01, 14+2k:b10, 14+2k+1:b11
            # W indices: 4k:c00, 4k+1:c10, 4k+2:c01, 4k+3:c11
            
            if t == 0:
                # P1 = (a+f)(b00+b11)
                U[0][col] = 1
                U[3][col] = 1
                V[2*k][col] = 1
                V[14+2*k+1][col] = 1
                W[4*k][col] = 1
                W[4*k+3][col] = 1
            elif t == 1:
                # P2 = (e+f)b01
                U[2][col] = 1
                U[3][col] = 1
                V[2*k+1][col] = 1
                W[4*k+1][col] = 1
                W[4*k+3][col] = -1
            elif t == 2:
                # P3 = a(b10-b11)
                U[0][col] = 1
                V[14+2*k][col] = 1
                V[14+2*k+1][col] = -1
                W[4*k+2][col] = 1
                W[4*k+3][col] = 1
            elif t == 3:
                # P4 = f(b01-b00)
                U[3][col] = 1
                V[2*k+1][col] = 1
                V[2*k][col] = -1
                W[4*k][col] = 1
                W[4*k+2][col] = 1
            elif t == 4:
                # P5 = (a+c)b11
                U[0][col] = 1
                U[1][col] = 1
                V[14+2*k+1][col] = 1
                W[4*k][col] = -1
                W[4*k+2][col] = 1
            elif t == 5:
                # P6 = (e-a)(b00+b10)
                U[2][col] = 1
                U[0][col] = -1
                V[2*k][col] = 1
                V[14+2*k][col] = 1
                W[4*k+1][col] = 1
                W[4*k+3][col] = 1
            elif t == 6:
                # P7 = (c-f)(b01+b11)
                U[1][col] = 1
                U[3][col] = -1
                V[2*k+1][col] = 1
                V[14+2*k+1][col] = 1
                W[4*k][col] = 1
                W[4*k+3][col] = -1
                
    return {"U": U, "V": V, "W": W}
