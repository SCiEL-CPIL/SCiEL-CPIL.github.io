"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Strassen rank-7 decomposition for 2x2x2
    # u[r][i][j], v[r][j][k], w[r][k][i] for i,j,k in {0,1}
    # Using 1D indexing: u_idx = i*m+j, v_idx = j*p2+k, w_idx = k*n+i where p2=2
    strassen_u = [
        {0: 1, 3: 1},        # M1: A[0,0]+A[1,1]
        {2: 1, 3: 1},        # M2: A[1,0]+A[1,1]
        {0: 1},              # M3: A[0,0]
        {3: 1},              # M4: A[1,1]
        {0: 1, 1: 1},        # M5: A[0,0]+A[0,1]
        {2: 1, 0: -1},       # M6: A[1,0]-A[0,0]
        {1: 1, 3: -1},       # M7: A[0,1]-A[1,1]
    ]
    strassen_v = [
        {0: 1, 3: 1},        # M1: B[0,0]+B[1,1]
        {0: 1},              # M2: B[0,0]
        {1: 1, 3: -1},       # M3: B[0,1]-B[1,1]
        {2: 1, 0: -1},       # M4: B[1,0]-B[0,0]
        {3: 1},              # M5: B[1,1]
        {0: 1, 2: 1},        # M6: B[0,0]+B[1,0]
        {1: 1, 3: 1},        # M7: B[0,1]+B[1,1]
    ]
    strassen_w = [
        {0: 1, 3: 1},        # M1: C[0,0]+C[1,1] -> w[k*n+i]: (0,0)=0, (1,1)=3
        {1: 1},              # M2: C[1,0] -> w[0*2+1]=1
        {2: 1},              # M3: C[0,1] -> w[1*2+0]=2
        {0: 1, 1: 1},        # M4: C[0,0]+C[1,0] -> w[0]=0, w[1]=1
        {0: -1, 2: 1},       # M5: -C[0,0]+C[0,1] -> w[0]=-1, w[2]=1
        {3: 1},              # M6: C[1,1] -> w[3]=1
        {0: 1},              # M7: C[0,0] -> w[0]=1
    ]
    
    total_rank = 7 * 7 + 4  # 53
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * total_rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    # 7 groups of 2 columns
    for g in range(7):
        k0 = 2 * g
        for r in range(7):
            # U factor
            for idx, val in strassen_u[r].items():
                U[idx][col] += val
            # V factor: j*p+k where k is local (0 or 1) mapped to k0, k0+1
            for idx, val in strassen_v[r].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][col] += val
            # W factor: k*n+i where k is local mapped to global
            for idx, val in strassen_w[r].items():
                k_local = idx // n
                i = idx % n
                k_global = k0 + k_local
                W[k_global * n + i][col] += val
            col += 1
    
    # 1 group of 1 column (k=14)
    # Schoolbook for 2x2x1: rank 4
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 14][col] = 1
            W[14 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
