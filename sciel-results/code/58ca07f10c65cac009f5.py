import time
import numpy as np
from math import isqrt
import itertools

def construct():
    start_time = time.time()
    time_limit = 140.0

    def gen_powerful(L):
        """Generate all powerful numbers up to L."""
        ps = set()
        max_b = int(L ** (1/3)) + 1
        for b in range(1, max_b):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            if max_a < 1:
                continue
            a_vals = np.arange(1, max_a + 1)
            p_vals = (a_vals * a_vals) * b3
            ps.update(p_vals.tolist())
        return sorted(ps)

    def count_pairs(N, pow_list, pow_set):
        """Count pairs (a,b) with a<=b, a+b=N, both powerful."""
        pairs = []
        half = N // 2
        for a in pow_list:
            if a > half:
                break
            b = N - a
            if b in pow_set:
                pairs.append([a, b])
        return pairs

    # Phase 1: Generate candidate N using sum-of-two-squares structure
    # N with many prime factors ≡ 1 (mod 4) has many representations as sum of two squares
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113,
                     137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257,
                     269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397]

    candidates = set()

    # Products of k distinct primes ≡ 1 mod 4
    for k in range(2, 14):
        for combo in itertools.combinations(primes_1mod4[:30], k):
            N = 1
            for p in combo:
                N *= p
                if N > 10**18:
                    break
            if N <= 10**18:
                candidates.add(N)
            # Also try with exponent 2 on smallest prime
            N2 = combo[0] * combo[0]
            for p in combo[1:]:
                N2 *= p
                if N2 > 10**18:
                    break
            if N2 <= 10**18:
                candidates.add(N2)
            if len(candidates) > 20000:
                break
        if len(candidates) > 20000:
            break

    candidates = sorted(candidates)

    # Phase 2: Generate powerful numbers up to a limit
    # We need powerful numbers up to max_candidate/2 to check pairs
    max_cand = candidates[-1] if candidates else 10**18
    L_pow = min(isqrt(max_cand) + 100, 10**9)
    pow_list = gen_powerful(L_pow)
    pow_set = set(pow_list)
    M = len(pow_list)

    # Phase 3: Evaluate candidates
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1

    for N_cand in candidates:
        if N_cand > 10**18:
            continue
        pairs = count_pairs(N_cand, pow_list, pow_set)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_cand
            best_pairs = pairs
        if time.time() - start_time > time_limit * 0.7:
            break

    # Phase 4: Random search for better N
    np.random.seed(42)
    for _ in range(20000):
        i = np.random.randint(0, M)
        j = np.random.randint(0, M)
        N_rand = pow_list[i] + pow_list[j]
        if N_rand > 10**18:
            continue
        pairs = count_pairs(N_rand, pow_list, pow_set)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_rand
            best_pairs = pairs
        if time.time() - start_time > time_limit * 0.95:
            break

    return {'N': int(best_N), 'pairs': best_pairs}
