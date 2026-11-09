import time

def construct():
    n, m, p = 2, 4, 5
    
    # Rank 18 decomposition
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    col = 0
    # Part 1: 2x4x2 (cols 0,1)
    # Sub 1.1: A[0:2, 0:2] * B[0:2, 0:2]
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                gi = li
                gj = lj # 0,1
                U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
        for lj in range(2):
            for lk in range(2):
                gj = lj # 0,1
                gk = lk # 0,1
                V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
        for lk in range(2):
            for li in range(2):
                gk = lk # 0,1
                gi = li
                W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
        col += 1
        
    # Sub 1.2: A[0:2, 2:4] * B[2:4, 0:2]
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                gi = li
                gj = 2 + lj # 2,3
                U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
        for lj in range(2):
            for lk in range(2):
                gj = 2 + lj # 2,3
                gk = lk # 0,1
                V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
        for lk in range(2):
            for li in range(2):
                gk = lk # 0,1
                gi = li
                W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
        col += 1
        
    # Part 2: 2x4x3 (cols 2,3,4)
    # Sub 2.1: A[0:2, 0:2] * B[0:2, 2:5]
    # Rank 2: r=0 for i=0, r=1 for i=1
    for r in range(2):
        gi = r
        for lj in range(2):
            gj = lj # 0,1
            U[gi * m + gj][col] = 1
        for lj in range(2):
            for lk in range(3):
                gj = lj # 0,1
                gk = 2 + lk # 2,3,4
                V[gj * p + gk][col] = 1
        for lk in range(3):
            gk = 2 + lk # 2,3,4
            W[gk * n + gi][col] = 1
        col += 1
        
    # Sub 2.2: A[0:2, 2:4] * B[2:4, 2:5]
    for r in range(2):
        gi = r
        for lj in range(2):
            gj = 2 + lj # 2,3
            U[gi * m + gj][col] = 1
        for lj in range(2):
            for lk in range(3):
                gj = 2 + lj # 2,3
                gk = 2 + lk # 2,3,4
                V[gj * p + gk][col] = 1
        for lk in range(3):
            gk = 2 + lk # 2,3,4
            W[gk * n + gi][col] = 1
        col += 1
            
    return {"U": U, "V": V, "W": W}
