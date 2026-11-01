DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(num_blocks):
        # M1 = (a+d)(e+h)
        U[0][col] = 1
        U[3][col] = 1
        V[2*k][col] = 1
        V[10+2*k+1][col] = 1
        W[4*k][col] = 1
        W[4*k+3][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[2][col] = 1
        U[3][col] = 1
        V[2*k][col] = 1
        W[4*k+1][col] = 1
        col += 1
        
        # M3 = a(f-h)
        U[0][col] = 1
        V[2*k+1][col] = 1
        V[10+2*k][col] = -1
        W[4*k+2][col] = 1
        W[4*k+3][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[3][col] = 1
        V[10+2*k][col] = 1
        V[2*k][col] = -1
        W[4*k][col] = 1
        W[4*k+1][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[0][col] = 1
        U[1][col] = 1
        V[10+2*k+1][col] = 1
        W[4*k][col] = -1
        W[4*k+2][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2][col] = 1
        U[0][col] = -1
        V[2*k][col] = 1
        V[2*k+1][col] = 1
        W[4*k+3][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1][col] = 1
        U[3][col] = -1
        V[10+2*k][col] = 1
        V[10+2*k+1][col] = 1
        W[4*k][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
