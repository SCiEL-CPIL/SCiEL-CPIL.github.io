    def add_schoolbook_2x1x5(j_off):
        nonlocal col
        for i in range(2):
            for k in range(5):
                U[i * m + j_off][col] = 1
                V[j_off * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
