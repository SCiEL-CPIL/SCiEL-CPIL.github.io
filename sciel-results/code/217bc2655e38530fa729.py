    for i in range(n):
        for a in range(m):
            for k in range(p):
                # sum over rank of U[i*m+a, :] * V[a*p+k, :] * W[k*n+i, :]
                prod = U_np[i * m + a] * V_np[a * p + k] * W_np[k * n + i]
                s = int(np.sum(prod))
                if s != 1:
                    valid = False
