import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen for 2x2 blocks
    # 5 blocks of 2 columns each
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A: i in {0,1}, j in {0,1} -> i*m + j
    # a=0, b=1, c=2, d=3
    
    # Indices for B block t (cols 2t, 2t+1): j in {0,1}, k in {2t, 2t+1} -> j*p + k
    # e = B[0, 2t], f = B[0, 2t+1]
    # g = B[1, 2t], h = B[1, 2t+1]
    
    # Indices for C block t (cols 2t, 2t+1): i in {0,1}, k in {2t, 2t+1} -> k*n + i
    # C11 is at k=2t, i=0 -> 2t*n + 0
    # C12 is at k=2t+1, i=0 -> (2t+1)*n + 0
    # C21 is at k=2t, i=1 -> 2t*n + 1
    # C22 is at k=2t+1, i=1 -> (2t+1)*n + 1
    
    col = 0
    for t in range(num_blocks):
        # M1 = (a + d)(e + h)
        U[0][col] = 1 # a
        U[3][col] = 1 # d
        V[0*p + 2*t][col] = 1 # e
        V[1*p + 2*t + 1][col] = 1 # h
        # C11 += M1, C22 += M1
        W[2*t*n + 0][col] = 1
        W[(2*t+1)*n + 1][col] = 1
        col += 1
        
        # M2 = (c + d)e
        U[2][col] = 1 # c
        U[3][col] = 1 # d
        V[0*p + 2*t][col] = 1 # e
        # C21 += M2, C22 -= M2
        W[2*t*n + 1][col] = 1
        W[(2*t+1)*n + 1][col] = -1
        col += 1
        
        # M3 = a(f - h)
        U[0][col] = 1 # a
        V[0*p + 2*t + 1][col] = 1 # f
        V[1*p + 2*t + 1][col] = -1 # -h
        # C12 += M3, C22 += M3
        W[(2*t+1)*n + 0][col] = 1
        W[(2*t+1)*n + 1][col] = 1
        col += 1
        
        # M4 = d(g - e)
        U[3][col] = 1 # d
        V[1*p + 2*t][col] = 1 # g
        V[0*p + 2*t][col] = -1 # -e
        # C11 += M4, C21 += M4
        W[2*t*n + 0][col] = 1
        W[2*t*n + 1][col] = 1
        col += 1
        
        # M5 = (a + b)h
        U[0][col] = 1 # a
        U[1][col] = 1 # b
        V[1*p + 2*t + 1][col] = 1 # h
        # C11 -= M5, C12 += M5
        W[2*t*n + 0][col] = -1
        W[(2*t+1)*n + 0][col] = 1
        col += 1
        
        # M6 = (c - a)(e + f)
        U[2][col] = 1 # c
        U[0][col] = -1 # -a
        V[0*p + 2*t][col] = 1 # e
        V[0*p + 2*t + 1][col] = 1 # f
        # C22 += M6
        W[(2*t+1)*n + 1][col] = 1
        col += 1
        
        # M7 = (b - d)(g + h)
        U[1][col] = 1 # b
        U[3][col] = -1 # -d
        V[1*p + 2*t][col] = 1 # g
        V[1*p + 2*t + 1][col] = 1 # h
        # C11 += M7
        W[2*t*n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
