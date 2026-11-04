for i in range(n):
    for j in range(m):
        for k in range(p):
            u = U[i*m + j]
            v = V[j*p + k]
            w = W[k*n + i]
            val = np.sum(u * v * w)
            if abs(val - 1.0) > tol:
                return False, val
