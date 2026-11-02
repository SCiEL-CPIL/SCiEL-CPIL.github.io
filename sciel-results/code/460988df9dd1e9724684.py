for i in range(n):
    for j in range(m):
        for k in range(p):
            val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
            if abs(val - 1) > 1e-9:
                return False
