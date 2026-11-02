import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Block partitioning: j into [0,1,2] and [3,4], k into [0,1,2] and [3,4]
    # Blocks: (n, j1, k1), (n, j2, k1), (n, j1, k2), (n, j2, k2)
    # where j1 = [0,1,2] (size 3), j2 = [3,4] (size 2)
    #       k1 = [0,1,2] (size 3), k2 = [3,4] (size 2)
    
    j_groups = [[0,1,2], [3,4]]
    k_groups = [[0,1,2], [3,4]]
    
    rank = 0
    U_rows = []  # list of (row_index, value) for each term
    V_rows = []
    W_rows = []
    
    for jg in j_groups:
        for kg in k_groups:
            nj = len(jg)
            nk = len(kg)
            for i in range(n):
                for jj in range(nj):
                    for kk in range(nk):
                        j = jg[jj]
                        k = kg[kk]
                        col = rank
                        U_rows.append((i*m + j, col, 1))
                        V_rows.append((j*p + k, col, 1))
                        W_rows.append((k*n + i, col, 1))
                        rank += 1
    
    # Build matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for row, col, val in U_rows:
        U[row][col] = val
    for row, col, val in V_rows:
        V[row][col] = val
    for row, col, val in W_rows:
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
