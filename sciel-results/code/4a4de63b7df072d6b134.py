    U2 = np.array([[round(u * 2) for u in row] for row in U], dtype=float)
    V2 = np.array([[round(v * 2) for v in row] for row in V], dtype=float)
    W2 = np.array([[round(w * 2) for w in row] for row in W], dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U2[i * m + j, r] * V2[j * p + k, r] * W2[k * n + i, r]
                if abs(s - 8.0) > tol:
                    return False
