def _construct_3x4x8():
    """Construct a rank-24 decomposition using row-wise 1x4x4 blocks."""
    n, m, p = 3, 4, 8
    rank = n * 2 * 4  # 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for cg in range(2):
            k_start = cg * 4
            for c_local in range(4):
                c = col
                for j in range(m):
                    if j == c_local:
                        U[i * m + j][c] = 1
                        for k in range(k_start, k_start + 4):
                            V[j * p + k][c] = 1
                for k in range(k_start, k_start + 4):
                    W[k * n + i][c] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
