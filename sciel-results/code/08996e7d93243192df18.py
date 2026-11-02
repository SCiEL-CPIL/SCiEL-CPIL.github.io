import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 multiplication
    # u, v, w are lists of 7 vectors
    # u[s] is vector of size 4 (i*m+j)
    # v[s] is vector of size 4 (j*p+k) for p=2
    # w[s] is vector of size 4 (k*n+i) for n=2
    
    # u indices: 00, 01, 10, 11
    # v indices: 00, 01, 10, 11
    # w indices: 00, 10, 01, 11 (k*n+i: 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3)
    # Wait, w index mapping:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w = [
        [1, 0, 0, 1], # M1 -> C00(0), C11(3)
        [0, 1, 0, 0], # M2 -> C10(1)
        [0, 0, 1, 1], # M3 -> C01(2), C11(3)
        [1, 1, 0, 0], # M4 -> C00(0), C10(1)
        [-1, 0, 1, 0],# M5 -> C00(0), C01(2)
        [0, 0, 0, 1], # M6 -> C11(3)
        [1, 0, 0, 0]  # M7 -> C00(0)
    ]
    
    # Block 1: k=0,1
    for s in range(7):
        col = s
        for ij in range(4):
            U[ij][col] = u[s][ij]
        for jk in range(4):
            # v index jk corresponds to j, k' where k' in {0,1}
            # V index is j*p + k. Here p=4.
            # For block 1, k is 0 or 1.
            # jk index in v[s] is j*2 + k'.
            # So j = jk // 2, k' = jk % 2.
            j = jk // 2
            k = jk % 2
            V[j * p + k][col] = v[s][jk]
        for ki in range(4):
            # w index ki corresponds to k', i.
            # W index is k*n + i. Here n=2.
            # For block 1, k is 0 or 1.
            # ki index in w[s] is k'*2 + i.
            # So k' = ki // 2, i = ki % 2.
            k = ki // 2
            i = ki % 2
            W[k * n + i][col] = w[s][ki]
            
    # Block 2: k=2,3
    for s in range(7):
        col = 7 + s
        for ij in range(4):
            U[ij][col] = u[s][ij]
        for jk in range(4):
            j = jk // 2
            k_prime = jk % 2
            k = 2 + k_prime
            V[j * p + k][col] = v[s][jk]
        for ki in range(4):
            k_prime = ki // 2
            i = ki % 2
            k = 2 + k_prime
            W[k * n + i][col] = w[s][ki]
            
    return {"U": U, "V": V, "W": W}
