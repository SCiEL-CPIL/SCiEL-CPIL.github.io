import time

def construct():
    deadline = time.time() + 85.0

    N = 1000000
    # Sieve for smallest prime factor up to N
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def lpf(x):
        if x <= 1:
            return 1
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p
            while x % p == 0:
                x //= p
        return max_pf

    best_n = 10000
    best_val = max(lpf(10000), lpf(10001))

    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        ln = lpf(n)
        if ln >= best_val:
            continue
        ln1 = lpf(n + 1)
        val = max(ln, ln1)
        if val < best_val:
            best_val = val
            best_n = n

    return best_n
