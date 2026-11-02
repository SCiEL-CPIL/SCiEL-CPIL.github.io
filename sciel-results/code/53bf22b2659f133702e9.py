                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            total2 = 0
                            for r in range(rank):
                                total2 += U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r]
                            if abs(total2) > 1e-10:
                                return False
