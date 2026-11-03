<<<<<<< SEARCH
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1
=======
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        if j_end - j_start == 1:
            j = j_start
            for i in range(i_start, i_end):
                U[i * m + j][col] = 1
            for k in range(k_start, k_end):
                V[j * p + k][col] = 1
                for i in range(i_start, i_end):
                    W[k * n + i][col] = 1
            col += 1
        else:
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        U[idx_u][col] = 1
                        V[idx_v][col] = 1
                        W[idx_w][col] = 1
                        col += 1
>>>>>>> REPLACE
