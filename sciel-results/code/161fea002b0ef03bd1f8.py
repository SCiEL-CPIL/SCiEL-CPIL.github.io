import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Helper to generate 4x4x2 decomposition
    def gen_4x4x2(k_offset):
        rank = 28
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for bi in range(2):
            for bj in range(2):
                for c in range(7):
                    col = (bi * 2 + bj) * 7 + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = k_offset + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    
                    for dk in range(2):
                        for di in range(2):
                            k = k_offset + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
        
        return U, V, W

    # Helper to generate 4x4x1 decomposition
    def gen_4x4x1(k_offset):
        rank = 4
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for r in range(4):
            # U_{ij, r} = delta_{j, r}
            for i in range(4):
                for j in range(4):
                    if j == r:
                        U[i * m + j][r] = 1
            
            # V_{jk, r} = delta_{j, k_offset} * delta_{k, k_offset}?
            # V is indexed by j*p + k.
            # We want S_B(r) = B_{r, k_offset}.
            # S_B(r) = sum_j B_{j, k_offset} V_{j, k_offset, r}.
            # So V_{r, k_offset, r} = 1.
            V[r * p + k_offset][r] = 1
            
            # W_{ik, r} = 1
            for i in range(4):
                W[k_offset * n + i][r] = 1
        
        return U, V, W

    # Combine parts
    # Part 1: 4x4x2 for k=0,1
    U1, V1, W1 = gen_4x4x2(0)
    # Part 2: 4x4x1 for k=2
    U2, V2, W2 = gen_4x4x1(2)
    # Part 3: 4x4x2 for k=3,4
    U3, V3, W3 = gen_4x4x2(3)
    # Part 4: 4x4x1 for k=5
    U4, V4, W4 = gen_4x4x1(5)
    
    total_rank = 28 + 4 + 28 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for U_part, V_part, W_part in [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3), (U4, V4, W4)]:
        r_part = len(U_part[0])
        for i in range(n * m):
            for c in range(r_part):
                U[i][offset + c] = U_part[i][c]
        for i in range(m * p):
            for c in range(r_part):
                V[i][offset + c] = V_part[i][c]
        for i in range(p * n):
            for c in range(r_part):
                W[i][offset + c] = W_part[i][c]
        offset += r_part
        
    return {"U": U, "V": V, "W": W}
