for i in range(n):
    for j in range(m):
        for k in range(p):
            total = 0
            for r in range(rank):
                total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            if total != 1:
                return False
