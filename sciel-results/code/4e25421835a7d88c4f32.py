DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    
    def build_rank32():
        rank = 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # p1 = (a+d)(e+h)
            U[0][col] = 1.0; U[3][col] = 1.0
            V[0*p+k0][col] = 1.0; V[1*p+k1][col] = 1.0
            W[k0*n+0][col] = 1.0; W[k0*n+1][col] = 1.0
            W[k1*n+0][col] = 1.0; W[k1*n+1][col] = 1.0
            col += 1
            
            # p2 = (c+d)e
            U[2][col] = 1.0; U[3][col] = 1.0
            V[0*p+k0][col] = 1.0
            W[k0*n+1][col] = 1.0; W[k1*n+1][col] = -1.0
            col += 1
            
            # p3 = a(f-h)
            U[0][col] = 1.0
            V[0*p+k1][col] = 1.0; V[1*p+k0][col] = -1.0
            W[k1*n+0][col] = 1.0; W[k1*n+1][col] = 1.0
            col += 1
            
            # p4 = d(g-e)
            U[3][col] = 1.0
            V[1*p+k0][col] = 1.0; V[0*p+k1][col] = -1.0
            W[k0*n+0][col] = 1.0; W[k0*n+1][col] = 1.0
            col += 1
            
            # p5 = (a+b)h
            U[0][col] = 1.0; U[1][col] = 1.0
            V[1*p+k1][col] = 1.0
            W[k0*n+0][col] = -1.0; W[k1*n+0][col] = 1.0
            col += 1
            
            # p6 = (c-a)(e+f)
            U[2][col] = 1.0; U[0][col] = -1.0
            V[0*p+k0][col] = 1.0; V[0*p+k1][col] = 1.0
            W[k1*n+1][col] = 1.0
            col += 1
            
            # p7 = (b-d)(g+h)
            U[1][col] = 1.0; U[3][col] = -1.0
            V[1*p+k0][col] = 1.0; V[1*p+k1][col] = 1.0
            W[k0*n+0][col] = 1.0; W[k0*n+1][col] = -1.0
            col += 1
        
        # Schoolbook for column 8
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1.0
                V[j*p+8][col] = 1.0
                W[8*n+i][col] = 1.0
                col += 1
        
        return U, V, W
    
    def verify(U, V, W):
        import numpy as np
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    U, V, W = build_rank32()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
