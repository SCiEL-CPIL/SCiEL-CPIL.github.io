def construct():
    import time
    start = time.time()
    DEADLINE = 115

    def time_check():
        return (time.time() - start) < DEADLINE

    n, m, p = 2, 4, 5
    from fractions import Fraction

    U_cols = []
    V_cols = []
    W_cols = []

    STRASSE = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    def add_2x4_x_4x2(a_row_map, b_col_map):
        nonlocal U_cols, V_cols, W_cols
        for part in range(2):
            a_rows = [part*2, part*2+1]
            b_cols = [part*2, part*2+1]
            for (X, Y, P) in STRASSE:
                u = [0]*8
                v = [0]*20
                w = [0]*10
                for (i, j), c in X.items():
                    r = a_rows[i]*4 + a_row_map[j]
                    u[r] += c
                for (a, b), c in Y.items():
                    r = a_row_map[a]*5 + b_col_map[b_cols[b]]
                    v[r] += c
                for (i, b), c in P.items():
                    r = b_col_map[b_cols[b]]*2 + a_rows[i]
                    w[r] += c
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)

    def add_2x4_x_4x1(a_row_map, b_col):
        nonlocal U_cols, V_cols, W_cols
        for j in range(4):
            for i in range(2):
                u = [0]*8
                v = [0]*20
                w = [0]*10
                u[i*4 + a_row_map[j]] = 1
                v[a_row_map[j]*5 + b_col] = 1
                w[b_col*2 + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)

    a_map = [0, 1, 2, 3]
    add_2x4_x_4x2(a_map, [0, 1])
    if time_check():
        add_2x4_x_4x2(a_map, [2, 3])
    if time_check():
        add_2x4_x_4x1(a_map, 4)

    rank = len(U_cols)

    def get_tensor_col(r):
        vec = [Fraction(0)] * (n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = Fraction(U_cols[r][i*m+j]) * Fraction(V_cols[r][j*p+k]) * Fraction(W_cols[r][k*n+i])
        return vec

    T = [get_tensor_col(r) for r in range(rank)]

    T_copy = [row[:] for row in T]
    pivot_cols = []
    curr_row = 0
    for col in range(rank):
        if not time_check():
            break
        if curr_row >= n*m*p:
            break
        pivot_row = -1
        for row in range(curr_row, n*m*p):
            if T_copy[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        T_copy[curr_row], T_copy[pivot_row] = T_copy[pivot_row], T_copy[curr_row]
        pivot_val = T_copy[curr_row][col]
        for row in range(n*m*p):
            if row != curr_row and T_copy[row][col] != 0:
                factor = T_copy[row][col] / pivot_val
                for j in range(col, rank):
                    T_copy[row][j] -= factor * T_copy[curr_row][j]
        pivot_cols.append(col)
        curr_row += 1

    if len(pivot_cols) < rank:
        U_cols = [U_cols[c] for c in pivot_cols]
        V_cols = [V_cols[c] for c in pivot_cols]
        W_cols = [W_cols[c] for c in pivot_cols]
        rank = len(pivot_cols)

    if time_check():
        T2 = [get_tensor_col(r) for r in range(rank)]
        T2_copy = [row[:] for row in T2]
        pivot_cols2 = []
        curr_row2 = 0
        for col in range(rank):
            if not time_check():
                break
            if curr_row2 >= n*m*p:
                break
            pivot_row2 = -1
            for row in range(curr_row2, n*m*p):
                if T2_copy[row][col] != 0:
                    pivot_row2 = row
                    break
            if pivot_row2 == -1:
                continue
            T2_copy[curr_row2], T2_copy[pivot_row2] = T2_copy[pivot_row2], T2_copy[curr_row2]
            pivot_val2 = T2_copy[curr_row2][col]
            for row in range(n*m*p):
                if row != curr_row2 and T2_copy[row][col] != 0:
                    factor2 = T2_copy[row][col] / pivot_val2
                    for j in range(col, rank):
                        T2_copy[row][j] -= factor2 * T2_copy[curr_row2][j]
            pivot_cols2.append(col)
            curr_row2 += 1
        if len(pivot_cols2) < rank:
            U_cols = [U_cols[c] for c in pivot_cols2]
            V_cols = [V_cols[c] for c in pivot_cols2]
            W_cols = [W_cols[c] for c in pivot_cols2]
            rank = len(pivot_cols2)

    U_list = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_list = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_list = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]

    return {"U": U_list, "V": V_list, "W": W_list}
