"""Search for M <= 1e18 with many k such that k*sigma(k)=M, using SPF-based sigma."""
import math
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    # ---- SPF sieve up to N ----
    # We want N large enough to find M with many solutions.
    # From knowledge, N=200k gives 4-6 solutions. Let's go larger.
    # SPF sieve is O(N). Sigma computation via SPF is O(N * log N) worst case but fast.
    N = 5000000
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:  # prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # ---- compute sigma(k) using SPF ----
    sigma = [0] * (N + 1)
    sigma[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so p^e || m for e>=1
            # sigma(n) = sigma(m) * (1 + p + ... + p^e) / (1 + ... + p^(e-1)) ... 
            # Better: use the formula sigma(n) = sigma(m) + p^e * sigma(m/p^e) ... 
            # Actually, use: if p^a || n, sigma(n) = sigma(n/p^a) * (p^(a+1)-1)/(p-1)
            # Let's compute differently: 
            # Let n = p^a * r where p not divide r.
            # sigma(n) = sigma(r) * (p^(a+1)-1)/(p-1)
            # We can store for each n: its "prime power part" and the rest.
            # Simpler: use the recurrence.
            # sigma(n) = sigma(m) + p * sigma(m) - p * sigma(m/p) ... no.
            # Let's use: if m = p^a * r (p not|r), n = p^(a+1)*r
            # sigma(n) = sigma(r)*(p^(a+2)-1)/(p-1)
            # sigma(m) = sigma(r)*(p^(a+1)-1)/(p-1)
            # So sigma(n) = sigma(m) + p^(a+1)*sigma(r)
            # We need sigma(r) and p^(a+1). 
            # Alternative: store two arrays: spf and the "largest prime power dividing n".
            # Let me just use a direct formula with a helper.
            # For efficiency, precompute: for each n, find p=spf[n], and the highest power p^a || n.
            # Then sigma(n) = sigma(n/p^a) * (p^(a+1)-1)//(p-1)
            # But n/p^a might not have sigma computed yet if we go in order... 
            # Actually n/p^a < n, so it is computed. Good.
            pass
        # This is getting complex. Let me use the divisor-sum sieve instead, which is simpler.
        pass

    # Let me just use the divisor-sum sieve which is straightforward.
    # Re-do with divisor sieve.
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d

    # ---- group by M ----
    groups = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            groups[M].append(k)

    # ---- find best ----
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    for M, ks in groups.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            if best_count >= 1000:
                break

    # ---- verify ----
    def factorize(n):
        out = {}
        d = 2
        while d * d <= n:
            if n % d == 0:
                c = 0
                while n % d == 0:
                    n //= d
                    c += 1
                out[d] = c
            d += 1 if d == 2 else 2
        if n > 1:
            out[n] = 1
        return out

    def sigma_exact(k):
        f = factorize(k)
        s = 1
        for p, e in f.items():
            s *= (p ** (e + 1) - 1) // (p - 1)
        return s

    if time.time() < DEADLINE - 3.0:
        verified = []
        for k in best_k:
            if k > math.isqrt(best_M):
                continue
            if k * sigma_exact(k) == best_M:
                verified.append(k)
        if verified:
            best_k = verified
    else:
        best_k = [k for k in best_k if k <= math.isqrt(best_M)]

    best_k = sorted(set(best_k))[:1000]
    if not best_k:
        return {'M': 333312, 'k': [336, 372, 434]}
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
