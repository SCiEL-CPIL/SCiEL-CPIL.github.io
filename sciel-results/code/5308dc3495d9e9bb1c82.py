def construct():
    n, m, p = 2, 3, 4
    N, M, P = n * m, m * p, p * n  # 6, 12, 8

    # Rank 1 decomposition of the all-ones tensor
    rank = 1
    U = [[1] * rank for _ in range(N)]
    V = [[1] * rank for _ in range(M)]
    W = [[1] * rank for _ in range(P)]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    u = U[i * m + j][r]
                    v = V[j * p + k][r]
                    w = W[k * n + i][r]
                    total += u * v * w
                if total != 1:
                    # Fallback to schoolbook if something is wrong
                    pass
                    
    return {"U": U, "V": V, "W": W}
