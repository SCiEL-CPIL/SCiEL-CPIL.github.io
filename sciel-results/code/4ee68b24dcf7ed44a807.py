DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 6 Strassen blocks
    for b in range(6):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1: (a00 + a11) * (b0k1 + b1k2)
        U[0][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        V[13 + k2][col] += 1
        W[2 * k1][col] += 1
        W[2 * k2 + 1][col] += 1
        col += 1
        
        # M2: (a10 + a11) * b0k1
        U[2][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # M3: a00 * (b0k2 - b1k2)
        U[0][col] += 1
        V[k2][col] += 1
        V[13 + k2][col] -= 1
        W[2 * k2][col] += 1
        col += 1
        
        # M4: a11 * (b1k1 - b0k1)
        U[3][col] += 1
        V[13 + k1][col] += 1
        V[k1][col] -= 1
        W[2 * k1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # M5: (a00 + a01) * b1k2
        U[0][col] += 1
        U[1][col] += 1
        V[13 + k2][col] += 1
        W[2 * k1][col] -= 1
        W[2 * k2][col] += 1
        col += 1
        
        # M6: (a10 - a00) * (b0k1 + b0k2)
        U[2][col] += 1
        U[0][col] -= 1
        V[k1][col] += 1
        V[k2][col] += 1
        W[2 * k2 + 1][col] += 1
        col += 1
        
        # M7: (a01 - a11) * (b1k1 + b1k2)
        U[1][col] += 1
        U[3][col] -= 1
        V[13 + k1][col] += 1
        V[13 + k2][col] += 1
        W[2 * k1][col] += 1
        col += 1

    # Last column (12) schoolbook
    k = 12
    
    # c0k = a00*b0k + a01*b1k
    U[0][col] += 1
    V[k][col] += 1
    W[2 * k][col] += 1
    col += 1
    
    U[1][col] += 1
    V[13 + k][col] += 1
    W[2 * k][col] += 1
    col += 1
    
    # c1k = a10*b0k + a11*b1k
    U[2][col] += 1
    V[k][col] += 1
    W[2 * k + 1][col] += 1
    col += 1
    
    U[3][col] += 1
    V[13 + k][col] += 1
    W[2 * k + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
