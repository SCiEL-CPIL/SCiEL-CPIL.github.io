def make_rank16():
    n, m, p = 2, 3, 5
    rank = 16
    U = [[0]*rank for _ in range(n*m)]  # 6 x 16
    V = [[0]*rank for _ in range(m*p)]  # 15 x 16
    W = [[0]*rank for _ in range(p*n)]  # 10 x 16
    
    r = 0
    # Term 0: j=2, all k
    U[0*3+2][r] = 1  # i=0, j=2 -> index 2
    U[1*3+2][r] = 1  # i=1, j=2 -> index 5
    V[2*5+0][r] = 1  # j=2, k=0 -> index 10
    V[2*5+1][r] = 1
    V[2*5+2][r] = 1
    V[2*5+3][r] = 1
    V[2*5+4][r] = 1
    W[0*2+0][r] = 1  # k=0, i=0
    W[0*2+1][r] = 1
    W[1*2+0][r] = 1
    W[1*2+1][r] = 1
    W[2*2+0][r] = 1
    W[2*2+1][r] = 1
    W[3*2+0][r] = 1
    W[3*2+1][r] = 1
    W[4*2+0][r] = 1
    W[4*2+1][r] = 1
    r += 1
