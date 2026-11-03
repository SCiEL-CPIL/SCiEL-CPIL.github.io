    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected value: 1
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a] * V_np[b] * W_np[c])
                if abs(val - 1) > 1e-9:
                    return False
