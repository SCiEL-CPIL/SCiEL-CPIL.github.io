DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    rank = 6 * 7 + 4  # 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    for block in range(6):
        c0 = 2 * block
        c1 = 2 * block + 1
        # e=B[0,c0], f=B[0,c1], g=B[1,c0], h=B[1,c1]
        # V indices: e→c0, f→c1, g→13+c0, h→13+c1
        # W indices: C[0,c0]→2*c0, C[0,c1]→2*c1, C[1,c0]→2*c0+1, C[1,c1]→2*c1+1
        
        # M1 = (a+d)(e+h): C[0,0]+1, C[1,1]+1
        U[0][col]=1; U[3][col]=1
        V[c0][col]=1; V[13+c1][col]=1
        W[2*c0][col]=1; W[2*c1+1][col]=1
        col+=1
        
        # M2 = (c+d)e: C[1,0]+1, C[1,1]-1
        U[2][col]=1; U[3][col]=1
        V[c0][col]=1
        W[2*c0+1][col]=1; W[2*c1+1][col]=-1
        col+=1
        
        # M3 = a(f-h): C[0,1]+1, C[1,1]+1
        U[0][col]=1
        V[c1][col]=1; V[13+c1][col]=-1
        W[2*c1][col]=1; W[2*c1+1][col]=1
        col+=1
        
        # M4 = d(g-e): C[0,0]+1, C[1,0]+1
        U[3][col]=1
        V[13+c0][col]=1; V[c0][col]=-1
        W[2*c0][col]=1; W[2*c0+1][col]=1
        col+=1
        
        # M5 = (a+b)h: C[0,0]-1, C[0,1]+1
        U[0][col]=1; U[1][col]=1
        V[13+c1][col]=1
        W[2*c0][col]=-1; W[2*c1][col]=1
        col+=1
        
        # M6 = (c-a)(e+f): C[1,1]+1
        U[2][col]=1; U[0][col]=-1
        V[c0][col]=1; V[c1][col]=1
        W[2*c1+1][col]=1
        col+=1
        
        # M7 = (b-d)(g+h): C[0,0]+1
        U[1][col]=1; U[3][col]=-1
        V[13+c0][col]=1; V[13+c1][col]=1
        W[2*c0][col]=1
        col+=1
    
    # Last column c=12
    c = 12
    # C[0,c] = a*B[0,c] + b*B[1,c]
    # C[1,c] = c*B[0,c] + d*B[1,c]
    
    U[0][col]=1; V[c][col]=1; W[2*c][col]=1; col+=1
    U[1][col]=1; V[13+c][col]=1; W[2*c][col]=1; col+=1
    U[2][col]=1; V[c][col]=1; W[2*c+1][col]=1; col+=1
    U[3][col]=1; V[13+c][col]=1; W[2*c+1][col]=1; col+=1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
