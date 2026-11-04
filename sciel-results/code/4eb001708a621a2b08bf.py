def construct():
    n, m, p = 2, 4, 5
    rank = 32
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2_2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i, start_col):
        terms = [
            ({(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1}),
            ({(0,0): 1}, {(0,0): 1, (0,1): -1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}),
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1, (0,1): -1}),
            ({(0,1): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1})
        ]
        
        for u, v, w in terms:
            for (i, j), val in u.items():
                U[(a_base_i + i) * m + (a_base_j + j)][start_col] += val
            for (j, k), val in v.items():
                V[(b_base_j + j) * p + (b_base_k + k)][start_col] += val
            for (k, i), val in w.items():
                W[(c_base_k + k) * n + (c_base_i + i)][start_col] += val
            start_col += 1
        return start_col

    def add_trivial_2x2_2x1(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i, start_col):
        # Term 1: A00 B00 -> C00
        U[a_base_i * m + a_base_j][start_col] = 1
        V[b_base_j * p + b_base_k][start_col] = 1
        W[c_base_k * n + c_base_i][start_col] = 1
        start_col += 1
        # Term 2: A01 B10 -> C00
        U[a_base_i * m + a_base_j + 1][start_col] = 1
        V[(b_base_j + 1) * p + b_base_k][start_col] = 1
        W[c_base_k * n + c_base_i][start_col] = 1
        start_col += 1
        # Term 3: A10 B00 -> C10
        U[(a_base_i + 1) * m + a_base_j][start_col] = 1
        V[b_base_j * p + b_base_k][start_col] = 1
        W[c_base_k * n + c_base_i + 1][start_col] = 1
        start_col += 1
        # Term 4: A11 B10 -> C10
        U[(a_base_i + 1) * m + a_base_j + 1][start_col] = 1
        V[(b_base_j + 1) * p + b_base_k][start_col] = 1
        W[c_base_k * n + c_base_i + 1][start_col] = 1
        start_col += 1
        return start_col

    # Block 1: A cols 0,1, B rows 0,1
    col = add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 0, 0, 2, 2, 0, col)
    col = add_trivial_2x2_2x1(0, 0, 0, 4, 4, 0, col)
    
    # Block 2: A cols 2,3, B rows 2,3
    col = add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 2, 2, 2, 2, 0, col)
    col = add_trivial_2x2_2x1(0, 2, 2, 4, 4, 0, col)
    
    return {"U": U, "V": V, "W": W}
