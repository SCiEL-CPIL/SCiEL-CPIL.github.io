        U_fb = np.zeros((n*n, n*n*n), dtype=float)
        V_fb = np.zeros((n*n, n*n*n), dtype=float)
        W_fb = np.zeros((n*n, n*n*n), dtype=float)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    r = i*n*n + j*n + k
                    U_fb[i*n + k, r] = 1.0
                    V_fb[k*n + j, r] = 1.0
                    W_fb[j*n + i, r] = 1.0
