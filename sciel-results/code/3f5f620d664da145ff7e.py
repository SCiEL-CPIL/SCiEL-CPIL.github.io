for i in range(n):
    for j in range(m):
        for k in range(p):
            u_val = U[i * m + j]
            v_val = V[j * p + k]
            w_val = W[k * n + i]
            s = np.sum(u_val * v_val * w_val)
            if abs(s - 1.0) > tol:
                return False
