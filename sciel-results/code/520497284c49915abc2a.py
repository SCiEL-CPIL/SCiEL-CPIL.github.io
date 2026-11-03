def _check(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if i == k else 0
