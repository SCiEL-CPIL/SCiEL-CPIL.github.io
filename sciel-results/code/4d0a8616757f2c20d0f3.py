    def add_term(u, v, w, idx):
        for i in range(n):
            for j in range(m):
                U[i * m + j][idx] += u[i][j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][idx] += v[j][k]
        for i in range(n):
            for k in range(p):
                W[k * n + i][idx] += w[i][k]
