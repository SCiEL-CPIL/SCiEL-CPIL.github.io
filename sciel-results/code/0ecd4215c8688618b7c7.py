"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 120s kill

    def time_exceeded():
        return time.time() >= DEADLINE

    def get_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                step = i
                st = i * i
                sieve[st::step] = bytearray(len(sieve[st::step]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Powerful numbers up to L
    L = 10**9
    primes = get_primes(int(math.isqrt(L)) + 1)

    def generate_powerful():
        pn_set = {1}
        stack = [(0, 1)]
        npr = len(primes)
        while stack:
            idx, val = stack.pop()
            if val > L:
                continue
            if val > 1:
                pn_set.add(val)
            for i in range(idx, npr):
                p = primes[i]
                nv = val * p * p
                if nv > L:
                    break
                pn_set.add(nv)
                stack.append((i + 1, nv))
                while nv * p <= L:
                    nv *= p
                    pn_set.add(nv)
                    stack.append((i + 1, nv))
                    if time_exceeded():
                        return pn_set
        return pn_set

    powerful_set = generate_powerful()

    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}

    pn_list = sorted(powerful_set)
    pn_set = set(pn_list)
    n_pn = len(pn_list)

    # Candidate N: high-divisor numbers in [10^8, 10^9]
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    candidates = set()
    N_lo = 10**8
    N_hi = 10**9

    # DFS over exponents of small primes to build N near [N_lo, N_hi]
    def gen_candidates(idx, val):
        if time_exceeded():
            return
        if idx == len(small_primes):
            if N_lo <= val <= N_hi:
                candidates.add(val)
            return
        p = small_primes[idx]
        v = val
        # exponent 0
        gen_candidates(idx + 1, val)
        # exponent >= 1
        v = val
        for e in range(1, 30):
            v *= p
            if v > N_hi * 2:
                break
            gen_candidates(idx + 1, v)

    gen_candidates(0, 1)

    if not candidates:
        candidates = {N_lo}

    # Score candidates by divisor count
    def divisor_count(n):
        cnt = 1
        m = n
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                cnt *= (e + 1)
        if m > 1:
            cnt *= 2
        return cnt

    cand_list = sorted(candidates, key=divisor_count, reverse=True)
    # Take top candidates
    top_cands = cand_list[:300]

    # For each candidate, count pairs
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1

    pn_arr = np.array(pn_list, dtype=np.int64)
    # Only powerful numbers <= N_hi (all are <= L = 10^9)
    # For counting pairs for N: a in pn_list with a <= N//2 and N-a in pn_set

    for N in top_cands:
        if time_exceeded():
            break
        if N < 2:
            continue
        # Count pairs a <= b, a+b=N, both powerful
        # a ranges over pn_list where a <= N//2
        half = N // 2
        # find index where pn_list > half
        idx_end = bisect.bisect_right(pn_list, half)
        count = 0
        pairs = []
        for i in range(idx_end):
            a = pn_list[i]
            b = N - a
            if b < a:
                break
            if b in pn_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs

    if best_count <= 1:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
