for i in range(n):
    for j in range(m):
        for k in range(p):
            expected = 1.0 if i == k else 0.0
            if abs(T[i, j, k] - expected) > tol:
                return False
