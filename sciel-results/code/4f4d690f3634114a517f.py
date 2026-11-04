    for a in range(n):
        for b in range(m):
            for c in range(p):
                T_target[a * m + b, b * p + c, c * n + a] = 1.0
