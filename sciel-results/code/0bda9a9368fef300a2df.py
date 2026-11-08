    def _verify(U, V, W, tol=1e-9):
        n, m, p = 2, 3, 3
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    