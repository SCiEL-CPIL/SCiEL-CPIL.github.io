    for i in range(n):
        for j in range(m):
            for k in range(p):
                k_div2 = k // 2
                if k % 2 == 0:
                    c = i * offset + j * p_half + k_div2
                else:
                    c = i * offset + j * p_half + k_div2 + offset
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
