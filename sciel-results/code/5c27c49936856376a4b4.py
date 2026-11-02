for i in range(n):
    for j in range(m):
        for k in range(p):
            total = 0
            for r in range(len(U[0])):
                total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            if abs(total - 1) > 1e-9:
                return False
