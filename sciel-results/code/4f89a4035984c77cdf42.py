"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-63 decomposition of 6x6 matrix multiplication using
    Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 9).
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication
    # U2 (4x7), V2 (4x7), W2 (4x7)
    # Row indexing for 2x2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1 = (a+b)(e+f) -> result[0,0]
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1
    
    # m2 = (c+d)e -> result[1,0]
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # m3 = a(f-h) -> result[0,1]
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # m4 = d(g-e) -> result[1,1]
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[3][3] = 1
    
    # m5 = (a+d)(e+h) -> result[0,1] and result[1,0]
    U2[0][4] = 1; U2[3][4] = 1
    V2[0][4] = 1; V2[3][4] = 1
    W2[1][4] = 1; W2[2][4] = 1
    
    # m6 = (b-d)(g+h) -> result[1,1]
    U2[1][5] = 1; U2[3][5] = -1
    V2[2][5] = 1; V2[3][5] = 1
    W2[3][5] = 1
    
    # m7 = (c-a)(f+h) -> result[0,0]
    U2[2][6] = 1; U2[0][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 3x3 schoolbook decomposition
    # U3 (9x9), V3 (9x9), W3 (9x9)
    # Row indexing for 3x3: (i,j) -> i*3+j
    
    U3 = [[0]*9 for _ in range(9)]
    V3 = [[0]*9 for _ in range(9)]
    W3 = [[0]*9 for _ in range(9)]
    
    col3 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+k][col3] = 1  # A_ik
                V3[k*3+j][col3] = 1  # B_kj
                W3[i*3+j][col3] = 1  # C_ij
                col3 += 1
    
    # Combine using Kronecker product structure
    rank = 9 * 7  # 63
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for block_term in range(9):
        for s_term in range(7):
            global_term = block_term * 7 + s_term
            
            # U: first matrix A
            for block_row in range(9):
                u3_val = U3[block_row][block_term]
                if u3_val == 0:
                    continue
                for inner_row in range(4):
                    u2_val = U2[inner_row][s_term]
                    if u2_val == 0:
                        continue
                    i = block_row // 3
                    k = block_row % 3
                    r = inner_row // 2
                    s = inner_row % 2
                    global_row = (i*2+r)*6 + (k*2+s)
                    U[global_row][global_term] += u3_val * u2_val
            
            # V: second matrix B
            for block_row in range(9):
                v3_val = V3[block_row][block_term]
                if v3_val == 0:
                    continue
                for inner_row in range(4):
                    v2_val = V2[inner_row][s_term]
                    if v2_val == 0:
                        continue
                    k = block_row // 3
                    j = block_row % 3
                    r = inner_row // 2
                    s = inner_row % 2
                    global_row = (k*2+r)*6 + (j*2+s)
                    V[global_row][global_term] += v3_val * v2_val
            
            # W: result matrix C
            for block_row in range(9):
                w3_val = W3[block_row][block_term]
                if w3_val == 0:
                    continue
                for inner_row in range(4):
                    w2_val = W2[inner_row][s_term]
                    if w2_val == 0:
                        continue
                    i = block_row // 3
                    j = block_row % 3
                    r = inner_row // 2
                    s = inner_row % 2
                    global_row = (i*2+r)*6 + (j*2+s)
                    W[global_row][global_term] += w3_val * w2_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
