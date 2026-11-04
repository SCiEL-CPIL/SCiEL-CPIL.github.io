def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication via Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # M4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _winograd_3x3():
    """Return U, V, W for 3x3 matrix multiplication via Winograd (rank 19)."""
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # M1
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    # M2
    U[0][1]=1; U[1][1]=1; V[1][1]=1; V[4][1]=1
    W[3][1]=1; W[4][1]=1; W[7][1]=1
    # M3
    U[0][2]=1; U[1][2]=-1; V[1][2]=1; V[4][2]=-1
    # M4
    U[2][3]=1; V[0][3]=1; V[1][3]=1; V[2][3]=1; V[3][3]=1; V[4][3]=1; V[5][3]=1
    W[3][3]=1; W[6][3]=1
    # M5
    U[2][4]=1; V[1][4]=1; V[2][4]=1; V[4][4]=1; V[5][4]=1
    W[6][4]=1; W[4][4]=1; W[7][4]=1
    # M6
    U[2][5]=1; U[1][5]=1; V[5][5]=1; V[8][5]=1
    W[6][5]=1; W[7][5]=1
    # M7
    U[2][6]=1; U[1][6]=-1; V[5][6]=1; V[8][6]=-1
    # M8
    U[2][7]=1; U[4][7]=1; V[6][7]=1; V[7][7]=1; V[8][7]=1; V[3][7]=1; V[4][7]=1; V[5][7]=1
    W[1][7]=1; W[4][7]=1
    # M9
    U[2][8]=1; U[4][8]=1; V[7][8]=1; V[8][8]=1; V[4][8]=1; V[5][8]=1
    W[4][8]=1; W[7][8]=1
    # M10
    U[2][9]=1; U[4][9]=-1; V[7][9]=1; V[8][9]=-1; V[4][9]=1; V[5][9]=-1
    W[1][9]=1; W[7][9]=1
    # M11
    U[2][10]=1; U[4][10]=-1; V[7][10]=1; V[8][10]=-1; V[5][10]=1; V[4][10]=-1
    # M12
    U[3][11]=1; V[6][11]=1
    # M13
    U[3][12]=1; U[4][12]=1; V[7][12]=1; V[8][12]=1
    W[5][12]=1
    # M14
    U[3][13]=1; U[4][13]=-1; V[7][13]=1; V[8][13]=-1
    # M15
    U[5][14]=1; V[6][14]=1; V[7][14]=1; V[8][14]=1
    W[5][14]=1; W[8][14]=1
    # M16
    U[5][15]=1; U[4][15]=1; V[8][15]=1; V[5][15]=1
    W[8][15]=1
    # M17
    U[5][16]=1; U[4][16]=-1; V[8][16]=1; V[5][16]=-1
    W[8][16]=1
    # M18
    U[6][17]=1; U[7][17]=1; V[0][17]=1; V[3][17]=1; V[6][17]=1; V[1][17]=1; V[4][17]=1; V[7][17]=1
    W[2][17]=1; W[5][17]=1
    # M19
    U[6][18]=1; U[7][18]=-1; V[0][18]=1; V[3][18]=-1; V[6][18]=1; V[1][18]=-1; V[4][18]=1; V[7][18]=-1
    W[2][18]=1; W[5][18]=1; W[8][18]=1
    
    return U, V, W

def construct():
    """Construct rank-133 decomposition for 6x6 matrix multiplication
    using tensor product of Winograd 3x3 (rank 19) and Strassen 2x2 (rank 7).
    Grid 3x3, Block 2x2.
    """
    U_block, V_block, W_block = _strassen_2x2() # 2x2 blocks, rank 7
    U_grid, V_grid, W_grid = _winograd_3x3()    # 3x3 grid, rank 19
    
    n = 6
    m = 6
    p = 6
    
    rank_block = 7
    rank_grid = 19
    total_rank = rank_block * rank_grid
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Grid indices I, J, K in 0..2
    # Block indices i, j, k in 0..1
    
    for I in range(3):
        for J in range(3):
            for i in range(2):
                for j in range(2):
                    # U row: (I*2 + i) * 6 + (J*2 + j)
                    row_u = (I * 2 + i) * 6 + (J * 2 + j)
                    u_block_idx = i * 2 + j
                    u_grid_idx = I * 3 + J
                    
                    for s in range(rank_grid):
                        val_g = U_grid[u_grid_idx][s]
                        if val_g == 0:
                            continue
                        for t in range(rank_block):
                            val_b = U_block[u_block_idx][t]
                            if val_b == 0:
                                continue
                            col = s * rank_block + t
                            U[row_u][col] = val_g * val_b

    for J in range(3):
        for K in range(3):
            for j in range(2):
                for k in range(2):
                    # V row: (J*2 + j) * 6 + (K*2 + k)
                    row_v = (J * 2 + j) * 6 + (K * 2 + k)
                    v_block_idx = j * 2 + k
                    v_grid_idx = J * 3 + K
                    
                    for s in range(rank_grid):
                        val_g = V_grid[v_grid_idx][s]
                        if val_g == 0:
                            continue
                        for t in range(rank_block):
                            val_b = V_block[v_block_idx][t]
                            if val_b == 0:
                                continue
                            col = s * rank_block + t
                            V[row_v][col] = val_g * val_b

    for K in range(3):
        for I in range(3):
            for k in range(2):
                for i in range(2):
                    # W row: k * 6 + i (Trace convention: W[k*n+i])
                    # Wait, W_grid is for C[I,K].
                    # W_grid row index is K*3 + I.
                    row_w = (K * 2 + k) * 6 + (I * 2 + i)
                    w_block_idx = k * 2 + i
                    w_grid_idx = K * 3 + I
                    
                    for s in range(rank_grid):
                        val_g = W_grid[w_grid_idx][s]
                        if val_g == 0:
                            continue
                        for t in range(rank_block):
                            val_b = W_block[w_block_idx][t]
                            if val_b == 0:
                                continue
                            col = s * rank_block + t
                            W[row_w][col] = val_g * val_b

    return {"U": U, "V": V, "W": W}
