def pollard_brent(n):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
    while True:
        y = random.randrange(1, n-1) + 1
        c = random.randrange(1, n-1) + 1
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y*y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = (y*y + c) % n
                    q = (q * abs(x-y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys*ys + c) % n
                g = math.gcd(abs(x-ys), n)
                if g > 1:
                    break
        if g != n:
            return g
