import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 44
    
    # Strassen factors for 2x2x2
    U_s = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: (a+b)
        [-1, 0, 1, 0],  # M6: (c-a)
        [0, 1, 0, -1],  # M7: (b-d)
    ]
    
    V_s = [
        [1, 1, 0, 0],   # M1: (e+h)
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: (f-h)
        [-1, 0, 1, 0],  # M4: (g-e)
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: (e+f)
        [0, 0, 1, 1],   # M7: (g+h)
    ]
    
    W_s = [
        [1, 0, 0, 1],   # M1: c00+=1, c11+=1
        [0, 0, 1, -1],  # M2: c10+=1, c11-=1
        [0, 1, 0, 1],   # M3: c01+=1, c11+=1
        [1, 0, 1, 0],   # M4: c00+=1, c10+=1
        [-1, 1, 0, 0],  # M5: c00-=1, c01+=1
        [0, 0, 0, 1],   # M6: c11+=1
        [1, 0, 0, 0],   # M7: c00+=1
    ]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(u_rows, v_rows, w_rows):
        nonlocal col
        for r in range(7):
            if time.time() > deadline:
                return
            for idx in range(4):
                U[u_rows[idx]][col] = U_s[r][idx]
                V[v_rows[idx]][col] = V_s[r][idx]
                W[w_rows[idx]][col] = W_s[r][idx]
            col += 1
    
    # W row indices for C[0:2,0:2]: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    w_rows_c02_02 = [0, 3, 1, 4]
    # W row indices for C[0:2,2:4]: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    w_rows_c02_24 = [6, 9, 7, 10]
    
    # Sub-part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], w_rows_c02_02)
    
    # Sub-part 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], w_rows_c02_24)
    
    # Sub-part 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], w_rows_c02_02)
    
    # Sub-part 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], w_rows_c02_24)
    
    # 1x4x4 part: row 2 of A, schoolbook (rank 16)
    for j in range(4):
        if time.time() > deadline:
            break
        for k in range(4):
            if time.time() > deadline:
                break
            u_row = 2 * 4 + j  # 8 + j
            v_row = j * 4 + k
            w_row = k * 3 + 2
            U[u_row][col] = 1.0
            V[v_row][col] = 1.0
            W[w_row][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
