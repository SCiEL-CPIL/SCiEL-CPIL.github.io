import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    max_m = 10**18

    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    n_primes = len(primes)

    m_to_ks = defaultdict(list)

    # Precompute prime powers and sigma contributions
    # Limit exponents such that p^e <= 10^9 (since k <= 10^9)
    # Actually k can be up to 10^9, but M <= 10^18.
    # k * sigma(k) <= 10^18.
    # sigma(k) >= k. So k^2 <= 10^18 => k <= 10^9.
    
    prime_powers = []
    for p in primes:
        powers = []
        p_pow = 1
        sigma_p = 1
        e = 0
        while True:
            # Check if adding this power keeps k within bounds?
            # We don't know k yet.
            # But p_pow itself must be <= 10^9.
            if p_pow > 10**9:
                break
            powers.append((p_pow, sigma_p))
            p_pow *= p
            if p_pow > 10**9:
                break
            sigma_p += p_pow
            e += 1
        prime_powers.append(powers)

    # Iterative DFS
    stack = [(0, 1, 1)]
    
    # Optimization: Local variables for speed
    pp = prime_powers
    mp = m_to_ks
    mm = max_m
    deadline_local = deadline
    time_check = time.time

    while stack:
        if time_check() > deadline_local:
            break

        idx, k, sigma_k = stack.pop()

        if idx == n_primes:
            M = k * sigma_k
            if M <= mm:
                mp[M].append(k)
            continue

        p = primes[idx]
        powers = pp[idx]

        # Iterate over powers of p
        for p_pow, sigma_p in powers:
            new_k = k * p_pow
            if new_k > 10**9: # k bound
                break

            new_sigma = sigma_k * sigma_p

            # Prune if M exceeds limit
            if new_k * new_sigma > mm:
                break

            stack.append((idx + 1, new_k, new_sigma))

    # Find the best M
    best_m = 0
    best_count = 0
    for M, ks in mp.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = mp[best_m]
    # Ensure distinct and within bounds
    best_ks = list(set(best_ks))
    # k <= sqrt(M) is always true for k >= 1, but check just in case
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
