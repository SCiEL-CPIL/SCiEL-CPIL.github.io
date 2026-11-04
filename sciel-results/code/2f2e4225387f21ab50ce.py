    for r1 in range(7):
        for ai in range(2):
            for ak in range(2):
                for aj in range(2):
                    s = ai * 4 + ak * 2 + aj
                    r = r1 * 4 + s
                    
                    for i1 in range(2):
                        for k1 in range(2):
                            for j1 in range(2):
                                i = 2 * i1 + ai
                                k = 2 * k1 + ak
                                j = 2 * j1 + aj
                                # ...
                                U[i * m + k][r] = U2[i1 * 2 + k1][r1]
                                V[k * p + j][r] = V2[k1 * 2 + j1][r1]
                                W[j * n + i][r] = W2[j1 * 2 + i1][r1]
