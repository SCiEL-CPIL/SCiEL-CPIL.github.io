import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # we need pp up to MAX_N + 1

    # Build smallest prime factor sieve
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute powerful parts using SPF
    pp_arr = [1] * (LIMIT + 1)
    for i in range(2, LIMIT + 1):
        n = i
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= (p ** e)
        pp_arr[i] = pp

    best_n = MIN_N
    best_product = 0

    for n in range(MIN_N, MAX_N):
        prod = pp_arr[n] * pp_arr[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n

    return best_n
