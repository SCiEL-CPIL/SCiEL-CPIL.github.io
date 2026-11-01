            # W side
            for i in range(n):
                for k in range(2):
                    W[(2 * t + k) * n + i][col] = strassen_W[i * 2 + k][s]
