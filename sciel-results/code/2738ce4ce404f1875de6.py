import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 10000000

    # Compute smallest prime factor for all n up to N using sieve
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i

    # Compute sigma(n) for all n using SPF
    sigma = np.ones(N + 1, dtype=np.int64)

    # Vectorized sigma computation: for each prime p, process multiples
    # sigma(n) = product over primes p^k || n of (1 + p + ... + p^k)
    # We build it incrementally: for each prime p, multiply sigma[n] by the
    # appropriate geometric factor for the power of p dividing n.
    
    # Alternative: use the formula sigma(n) = sum of divisors via the multiplicative property
    # For each prime p, for each power p^k, for each n = p^k * m where gcd(m,p)=1:
    #   sigma(n) = sigma(m) * (1 + p + ... + p^k)
    # This is complex to vectorize directly. Let's use a different approach.
    
    # Use the fact that sigma is multiplicative. We can compute it using the
    # additive formula: sigma(n) = sum_{d|n} d. But that's O(n log n).
    
    # Better: for each prime p, compute the contribution. 
    # For n = p^k * m where gcd(m, p) = 1:
    #   sigma(n) = sigma(m) * (p^(k+1) - 1) / (p - 1)
    
    # We process primes in order. For each prime p, we update sigma for all
    # numbers whose smallest prime factor is p (i.e., spf[n] == p).
    
    # Actually, let's use a cleaner approach:
    # sigma[1] = 1
    # For n from 2 to N:
    #   p = spf[n]
    #   m = n // p
    #   if m % p == 0:
    #       # p divides m, so n has p^(k+1) where m has p^k
    #       # sigma(n) = sigma(m) * (1 + p + ... + p^(k+1)) / (1 + p + ... + p^k) * ... 
    #       # This gets complicated. Let's store the highest power of spf dividing n.
    
    # Let's use a different strategy: store for each n the "remaining part" after
    # removing all factors of spf[n], and the geometric sum for that prime power.
    
    # rem[n] = n / p^k where p = spf[n] and p^k || n
    # geom[n] = 1 + p + p^2 + ... + p^k
    # sigma[n] = sigma[rem[n]] * geom[n]
    
    rem = np.zeros(N + 1, dtype=np.int32)
    geom = np.zeros(N + 1, dtype=np.int64)
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n
        k = 0
        while m % p == 0:
            m //= p
            k += 1
        rem[n] = m
        # Compute 1 + p + ... + p^k
        g = 1
        pp = 1
        for _ in range(k):
            pp *= p
            g += pp
        geom[n] = g
        sigma[n] = sigma[m] * g

    if time.time() > deadline - 10.0:
        return [11, 6]

    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)

    # Find best pair using two-pointer approach within each group
    best_error_num = 1  # error = 1/3
    best_error_den = 3
    best_pair = (11, 6)

    for s, lst in groups.items():
        if time.time() > deadline - 3.0:
            break
        if len(lst) < 2:
            continue
        lst.sort()
        n_list = len(lst)
        for i in range(n_list):
            b = lst[i]
            target_num = 3 * b
            # We want a such that a/b ≈ 3/2, i.e., 2a ≈ 3b
            # So a ≈ 3b/2
            # Binary search for the closest a to 3b/2 in lst
            target = target_num / 2
            pos = bisect.bisect_left(lst, target)
            for idx in [pos - 1, pos]:
                if 0 <= idx < n_list:
                    a = lst[idx]
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    # Compare num/den < best_error_num/best_error_den
                    # i.e., num * best_error_den < best_error_num * den
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
