def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        # Block b handles output columns k=2b and k=2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Part 1: A[2x1] * B[1x2]
        # A is col 2 (j=2). B is row 2 (j=2).
        # 4 terms:
        # i=0, j=2, k=k0
        U[0*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
        
        # i=0, j=2, k=k1
        U[0*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 0][col] = 1
        col += 1
        
        # i=1, j=2, k=k0
        U[1*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # i=1, j=2, k=k1
        U[1*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # Part 2: A[2x2] * B[2x2] using Strassen
        # A indices: (0,0), (0,1), (1,0), (1,1) -> U idx: 0, 1, 3, 4
        # B indices: (0,k0), (0,k1), (1,k0), (1,k1) -> V idx: k0, k1, 8+k0, 8+k1
        
        # M1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[8+k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[0][col] = 1
        V[k1][col] = 1
        V[8+k1][col] = -1
        W[k0*n + 0][col] = 1 # c00 gets M3? No. c00 = M1 + M4 - M5 + M7. M3 is in c01, c11.
        # Let's re-verify Strassen output mapping.
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So for M3:
        # c01 gets +1. c11 gets +1.
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[4][col] = 1
        V[8+k0][col] = 1
        V[k0][col] = -1
        # c00 gets +1. c10 gets +1.
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[8+k1][col] = 1
        # c00 gets -1. c01 gets +1.
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[3][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[k1][col] = 1
        # c11 gets +1.
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[1][col] = 1
        U[4][col] = -1
        V[8+k0][col] = 1
        V[8+k1][col] = 1
        # c00 gets +1.
        W[k0*n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
