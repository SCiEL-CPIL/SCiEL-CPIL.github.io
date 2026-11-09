for i in range(n):
    for j in range(m):
        for k in range(p):
            total = 0.0
            for c in range(rank):
                total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(total - 1.0) > 1e-9:
                return False
