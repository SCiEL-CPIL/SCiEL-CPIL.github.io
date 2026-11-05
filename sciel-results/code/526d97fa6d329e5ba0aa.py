import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    LIMIT = N + 1

    start_time = time.time()
    deadline = start_time + 55

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    if time.time() > deadline:
        return MIN_N

    # 2. Compute powerful part directly using multiplicative recurrence
    # pp[1] = 1; for i >= 2, let p = spf[i], m = i // p
    # if spf[m] == p, then v_p(i) = v_p(m) + 1; else v_p(i) = 1
    # pp[i] = pp[m] * (p if v_p(i) >= 2 else 1) ... but we need to be careful.
    # Actually: powerful part of i = product of p^e where e >= 2.
    # Let i = p^a * r where p doesn't divide r.
    # powerful_part(i) = p^a * powerful_part(r) if a >= 2, else powerful_part(r).
    # We can compute this iteratively.

    pp = [1] * (LIMIT + 1)
    pp[0] = 0
    pp[1] = 1

    for i in range(2, LIMIT + 1):
        p = spf[i]
        m = i // p
        if spf[m] == p:
            # p divides m, so v_p(i) >= 2
            # pp[i] should include p^v_p(i) times pp[r] where r is i with all p's removed
            # pp[m] already includes p^(v_p(m)) if v_p(m) >= 2, or just pp[r] if v_p(m) == 1
            # We need: pp[i] = pp[m] * p if v_p(m) >= 2, else pp[m] * p^2 if v_p(m) == 1
            # Hmm, this is getting complex. Let me use a different approach.
            pass

    # Alternative: compute pp using the fact that pp is multiplicative over prime powers.
    # For each i, find its full factorization using spf, but do it efficiently.
    # Since spf gives us the smallest prime factor, we can factor each i in O(log i) time.
    # But with LIMIT = 1000001, this is about 10^6 * ~20 = 20M operations, which is fine.

    # Let's just do direct factorization for each number.
    pp2 = [1] * (LIMIT + 1)
    pp2[0] = 0
    pp2[1] = 1

    for i in range(2, LIMIT + 1):
        x = i
        prod = 1
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt >= 2:
                prod *= p ** cnt
        pp2[i] = prod

    if time.time() > deadline:
        return MIN_N

    # 3. Find max product
    best_n = MIN_N
    best_val = 0

    for n in range(MIN_N, N + 1):
        if time.time() > deadline:
            break
        val = pp2[n] * pp2[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
