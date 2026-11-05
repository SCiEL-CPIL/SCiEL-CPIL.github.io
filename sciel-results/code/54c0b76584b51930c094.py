    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
    