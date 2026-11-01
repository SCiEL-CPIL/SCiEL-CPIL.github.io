    def add_schoolbook_221(A_rows, A_cols, B_row, B_col):
        nonlocal col
        for i in A_rows:
            for j in A_cols:
                u_idx = i*m + j
                v_idx = j*p + B_col
                w_idx = B_col*n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
