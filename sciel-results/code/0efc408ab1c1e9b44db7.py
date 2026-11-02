def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Term 0-2: diagonal for i=0,1,2
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # I'll use a known rank-23 decomposition
    # ...
