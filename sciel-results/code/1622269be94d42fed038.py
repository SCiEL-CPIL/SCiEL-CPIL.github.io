def verify(U, V, W, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for l in range(r):
                    val += U[i*m + j, l] * V[j*p + k, l] * W[k*n + i, l]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
