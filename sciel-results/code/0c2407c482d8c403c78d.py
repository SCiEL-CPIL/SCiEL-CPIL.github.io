def verify_decomposition(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(s - 1) > 1e-10:
                    return False, i, j, k, s
    return True, None, None, None, None
