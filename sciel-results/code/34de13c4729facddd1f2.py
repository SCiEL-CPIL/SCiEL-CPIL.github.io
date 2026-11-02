def construct():
    n, m, p = 2, 2, 5
    rank = 16
    
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(10)]
    W = [[0.0]*rank for _ in range(10)]
    
    # Term 0: A[0,0] * B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: A[0,0] * B[0,1] -> C[1,0]
    U[0][1] = 1; V[1][1] = 1; W[2][1] = 1
    # ... etc
