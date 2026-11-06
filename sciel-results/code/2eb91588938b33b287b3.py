import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    N = 1000000

    # pp[i] = powerful part of i (product of p^e for all prime powers p^e || i with e>=2)
    pp = [1] * (N + 1)

    # Sieve primes up to sqrt(N)
    limit = int(math.isqrt(N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            step = i
            start_sq = i * i
            for j in range(start_sq, limit + 1, step):
                is_prime[j] = False

    # Accumulate powerful parts via sieve over prime powers
    for p in primes:
        val = p * p
        while val <= N:
            next_val = val * p
            # For each multiple m of val, if m is NOT divisible by next_val,
            # then the exact power of p in m is the current val (not a higher power)
            # So we multiply pp[m] by val
            # m = val * k; m % next_val == 0 iff k % p == 0
            k = 1
            for m in range(val, N + 1, val):
                if k % p != 0:
                    pp[m] *= val
                k += 1
            val = next_val

    # Find the best n in [10000, 999999]
    best_n = 10000
    best_val = -1

    for n in range(10000, N):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
