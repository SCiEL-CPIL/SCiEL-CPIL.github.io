"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    max_m = 10**18

    # Strategy: Generate k values where k*sigma(k) <= max_m
    # Find M with the most such k values
    # Use efficient generation with good pruning

    # Use primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_primes = len(primes)

    m_to_ks = defaultdict(list)
    count = [0]

    # Precompute prime powers and sigma contributions for each prime
    # For prime p, we store: (p^e, sigma(p^e)) pairs
    prime_powers = []
    for p in primes:
        powers = []
        p_pow = 1
        sigma_p = 1
        for e in range(0, 60):
            powers.append((p_pow, sigma_p))
            p_pow *= p
            if p_pow > max_m:
                break
            sigma_p += p_pow
        prime_powers.append(powers)

    # Iterative DFS with explicit stack to avoid recursion depth issues
    # State: (idx, k, sigma_k)
    stack = [(0, 1, 1)]

    while stack:
        if time.time() > deadline:
            break

        idx, k, sigma_k = stack.pop()
        count[0] += 1

        if idx == n_primes:
            M = k * sigma_k
            if M <= max_m:
                m_to_ks[M].append(k)
            continue

        p = primes[idx]
        powers = prime_powers[idx]

        for p_pow, sigma_p in powers:
            new_k = k * p_pow
            if new_k > max_m:
                break

            new_sigma = sigma_k * sigma_p

            # Prune if M exceeds limit
            if new_k * new_sigma > max_m:
                break

            stack.append((idx + 1, new_k, new_sigma))

    # Find the best M
    best_m = 0
    best_count = 0
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = m_to_ks[best_m]
    # Ensure distinct and within bounds
    best_ks = list(set(best_ks))
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
# EVOLVE-BLOCK-END
