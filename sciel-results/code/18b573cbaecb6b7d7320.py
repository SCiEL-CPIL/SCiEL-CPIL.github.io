    def residual(x):
        U = x[:n*m*R].reshape((n*m, R))
        V = x[n*m*R:n*m*R + m*p*R].reshape((m*p, R))
        W = x[n*m*R + m*p*R:].reshape((p*n, R))
        
        res = np.zeros(n_res)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    val = np.einsum('r,r,r->', U[i*m+j], V[j*p+k], W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res
