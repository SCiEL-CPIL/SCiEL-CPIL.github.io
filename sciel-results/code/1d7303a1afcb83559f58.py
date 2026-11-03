import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, i_out_offset, k_out_offset):
        # U_block shape: (i_range * j_range, rank_block)
        # V_block shape: (j_range * k_range, rank_block)
        # W_block shape: (k_range * i_range, rank_block)
        # Indices in block are local.
        # Global U index: (i_offset + i_local) * m + (j_offset + j_local)
        # Global V index: (j_offset + j_local) * p + (k_offset + k_local)
        # Global W index: (k_out_offset + k_local) * n + (i_out_offset + i_local)
        pass

    # Block 1: Strassen 2x2x2
    # i in {0,1}, j in {0,1}, k in {0,1}
    # Output i' in {0,1}, k' in {0,1}
    # Rank 7
    # Local U: 2x2=4 rows, 7 cols
    # Local V: 2x2=4 rows, 7 cols
    # Local W: 2x2=4 rows, 7 cols
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # Strassen formulas
    # M1 = (a00+a11)(b00+b11)
    # U1: a00(0,0), a11(1,1)
    # V1: b00(0,0), b11(1,1)
    # W1: c00(0,0), c11(1,1)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a10+a11)b00
    # U1: a10(1,0), a11(1,1)
    # V1: b00(0,0)
    # W1: c10(1,0), c11(1,1)
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[2][1] = 1; W1[3][1] = -1 # c11 = M1 - M2 ...
    
    # Let's use the standard Strassen output combination
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # M1
    U1[0][0]=1; U1[3][0]=1
    V1[0][0]=1; V1[3][0]=1
    W1[0][0]=1; W1[3][0]=1
    
    # M2
    U1[2][1]=1; U1[3][1]=1
    V1[0][1]=1
    W1[2][1]=1; W1[3][1]=-1
    
    # M3
    U1[0][2]=1
    V1[1][2]=1; V1[3][2]=-1
    W1[1][2]=1; W1[3][2]=1
    
    # M4
    U1[3][3]=1
    V1[2][3]=1; V1[0][3]=-1
    W1[0][3]=1; W1[2][3]=1
    
    # M5
    U1[0][4]=1; U1[1][4]=1
    V1[3][4]=1
    W1[0][4]=-1; W1[1][4]=1
    
    # M6
    U1[2][5]=1; U1[0][5]=-1
    V1[0][5]=1; V1[1][5]=1
    W1[3][5]=1
    
    # M7
    U1[1][6]=1; U1[3][6]=-1
    V1[2][6]=1; V1[3][6]=1
    W1[0][6]=1
    
    # Add Block 1 to global
    for i_l in range(2):
        for j_l in range(2):
            u_idx = (0 + i_l) * 3 + (0 + j_l)
            for r in range(7):
                U[u_idx][col + r] += U1[i_l * 2 + j_l][r]
    for j_l in range(2):
        for k_l in range(2):
            v_idx = (0 + j_l) * 4 + (0 + k_l)
            for r in range(7):
                V[v_idx][col + r] += V1[j_l * 2 + k_l][r]
    for k_l in range(2):
        for i_l in range(2):
            w_idx = (0 + k_l) * 3 + (0 + i_l)
            for r in range(7):
                W[w_idx][col + r] += W1[k_l * 2 + i_l][r]
    col += 7
    
    # Block 2: Schoolbook 2x3x2
    # i in {0,1}, j in {0,1,2}, k in {2,3}
    # Output i' in {0,1}, k' in {2,3}
    # Rank 12
    # Local U: 2x3=6 rows
    # Local V: 3x2=6 rows
    # Local W: 2x2=4 rows
    
    for i_l in range(2):
        for j_l in range(3):
            for k_l in range(2):
                u_idx = (0 + i_l) * 3 + (0 + j_l)
                v_idx = (0 + j_l) * 4 + (2 + k_l)
                w_idx = (2 + k_l) * 3 + (0 + i_l)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    # col should be 7 + 12 = 19
    
    # Block 3: Schoolbook 1x3x2
    # i in {2}, j in {0,1,2}, k in {0,1}
    # Output i' in {2}, k' in {0,1}
    # Rank 6
    
    for i_l in range(1):
        for j_l in range(3):
            for k_l in range(2):
                u_idx = (2 + i_l) * 3 + (0 + j_l)
                v_idx = (0 + j_l) * 4 + (0 + k_l)
                w_idx = (0 + k_l) * 3 + (2 + i_l)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    # col should be 19 + 6 = 25
    
    # Block 4: Schoolbook 1x3x2
    # i in {2}, j in {0,1,2}, k in {2,3}
    # Output i' in {2}, k' in {2,3}
    # Rank 6
    
    for i_l in range(1):
        for j_l in range(3):
            for k_l in range(2):
                u_idx = (2 + i_l) * 3 + (0 + j_l)
                v_idx = (0 + j_l) * 4 + (2 + k_l)
                w_idx = (2 + k_l) * 3 + (2 + i_l)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    # col should be 25 + 6 = 31

    return {"U": U, "V": V, "W": W}
