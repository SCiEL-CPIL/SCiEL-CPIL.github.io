"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Approach:
- Fast tau_perp: generate all divisors sorted, then count adjacent coprime pairs.
- Search: best-first search over exponent vectors of the first ~16 primes,
  with moves being increment/decrement a single exponent (and swap between primes).
- Deadline guard: return best found before wall-clock limit.
"""
import math
import time
import heapq
from functools import lru_cache

MAX_N = 10**18
DEADLINE = 140.0  # safety margin before 150s kill

# First 16 primes (product of first 15 ~ 6e17, 16th * 53 ~ 3e19 too big for sq-free
# but we allow exponents 0..3 on larger primes)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
NPRIMES = len(PRIMES)


def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors.
    Returns (count, num_divisors)."""
    if n == 1:
        return 0, 1
    # Factorize n
    factors = []
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            factors.append((p, e))
        p += 1 if p == 2 else 2
    if m > 1:
        factors.append((m, 1))
    # Generate all divisors with their prime-support bitmask
    # divisors: list of (value, mask)
    divs = [(1, 0)]
    for p, e in factors:
        new_divs = []
        for (v, mask) in divs:
            for k in range(e + 1):
                new_divs.append((v * (p ** k), mask | (1 << len(factors)) if k > 0 else mask))
        # fix mask: bit i set if divisor is divisible by factors[i]'s prime
        # redo properly
        divs = []
        for (v, mask) in [(1, 0)]:
            pass
    # Redo divisor generation cleanly
    divs = [(1, 0)]
    for idx, (p, e) in enumerate(factors):
        new_divs = []
        bit = 1 << idx
        for (v, mask) in divs:
            pk = 1
            for k in range(e + 1):
                if k == 0:
                    new_divs.append((v, mask))
                else:
                    new_divs.append((v * pk, mask | bit))
                pk *= p
        divs = new_divs
    divs.sort(key=lambda x: x[0])
    count = 0
    for i in range(len(divs) - 1):
        a, ma = divs[i]
        b, mb = divs[i + 1]
        if ma & mb == 0:
            count += 1
    return count, len(divs)


def _tau_perp_from_exponents(exps):
    """Compute tau_perp from exponent vector (fast, no re-factorization)."""
    factors = [(PRIMES[i], exps[i]) for i in range(NPRIMES) if exps[i] > 0]
    if not factors:
        return 0
    # Generate all divisors with bitmask
    divs = [(1, 0)]
    for idx, (p, e) in enumerate(factors):
        new_divs = []
        bit = 1 << idx
        for (v, mask) in divs:
            pk = 1
            for k in range(e + 1):
                if k == 0:
                    new_divs.append((v, mask))
                else:
                    new_divs.append((v * pk, mask | bit))
                pk *= p
        divs = new_divs
    divs.sort(key=lambda x: x[0])
    count = 0
    for i in range(len(divs) - 1):
        if divs[i][1] & divs[i + 1][1] == 0:
            count += 1
    return count


def _exponents_to_n(exps):
    n = 1
    for i, e in enumerate(exps):
        if e > 0:
            n *= PRIMES[i] ** e
    return n


def construct():
    start_time = time.time()
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed fallback

    # Initial: evaluate seed
    s, _ = _tau_perp(best_n)
    best_score = s

    # Best-first search over exponent vectors
    # State: tuple of exponents for first K primes
    # We search over first 12 primes with exponents 0..3 (bounded)
    K = 12
    MAX_EXP = 3

    def valid_exps(exps):
        n = 1
        for i in range(K):
            if exps[i] > 0:
                n *= PRIMES[i] ** exps[i]
        return n <= MAX_N

    # Start from all-ones for first 15 primes if valid, else first 14
    init = [0] * K
    for i in range(K):
        init[i] = 1
    if not valid_exps(init):
        # reduce
        while not valid_exps(init):
            for i in range(K - 1, -1, -1):
                if init[i] > 1:
                    init[i] -= 1
                    break
                else:
                    init[i] = 0
                    break

    init_tup = tuple(init)
    s_init = _tau_perp_from_exponents(init_tup)
    if s_init > best_score:
        best_score = s_init
        best_n = _exponents_to_n(init_tup)

    # Priority queue: (-score, counter, exponents_tuple)
    counter = 0
    heap = [(-s_init, counter, init_tup)]
    visited = {init_tup}
    counter += 1

    # Also seed with the original 8-prime square-free
    orig = [1] * 8 + [0] * (K - 8)
    orig_tup = tuple(orig)
    if orig_tup not in visited:
        s_orig = _tau_perp_from_exponents(orig_tup)
        if s_orig > best_score:
            best_score = s_orig
            best_n = _exponents_to_n(orig_tup)
        heapq.heappush(heap, (-s_orig, counter, orig_tup))
        visited.add(orig_tup)
        counter += 1

    # Also try square-free with 13, 14 primes (first K primes, exponents 0/1)
    for npr in [10, 11, 12, 13, 14]:
        if npr > K:
            continue
        cand = [1] * npr + [0] * (K - npr)
        if valid_exps(cand):
            cand_tup = tuple(cand)
            if cand_tup not in visited:
                s_c = _tau_perp_from_exponents(cand_tup)
                if s_c > best_score:
                    best_score = s_c
                    best_n = _exponents_to_n(cand_tup)
                heapq.heappush(heap, (-s_c, counter, cand_tup))
                visited.add(cand_tup)
                counter += 1

    while heap:
        if time.time() - start_time > DEADLINE:
            break
        neg_s, _, exps = heapq.heappop(heap)
        cur_s = -neg_s

        # Generate neighbors: increment or decrement each exponent
        for i in range(K):
            for delta in (1, -1):
                new_e = exps[i] + delta
                if new_e < 0 or new_e > MAX_EXP:
                    continue
                new_exps = list(exps)
                new_exps[i] = new_e
                if not valid_exps(new_exps):
                    continue
                new_tup = tuple(new_exps)
                if new_tup in visited:
                    continue
                visited.add(new_tup)
                s_new = _tau_perp_from_exponents(new_tup)
                if s_new > best_score:
                    best_score = s_new
                    best_n = _exponents_to_n(new_tup)
                heapq.heappush(heap, (-s_new, counter, new_tup))
                counter += 1

        # Swap moves: decrement i, increment j (i != j)
        if time.time() - start_time > DEADLINE:
            break
        for i in range(K):
            if exps[i] == 0:
                continue
            for j in range(K):
                if i == j or exps[j] == MAX_EXP:
                    continue
                new_exps = list(exps)
                new_exps[i] -= 1
                new_exps[j] += 1
                if not valid_exps(new_exps):
                    continue
                new_tup = tuple(new_exps)
                if new_tup in visited:
                    continue
                visited.add(new_tup)
                s_new = _tau_perp_from_exponents(new_tup)
                if s_new > best_score:
                    best_score = s_new
                    best_n = _exponents_to_n(new_tup)
                heapq.heappush(heap, (-s_new, counter, new_tup))
                counter += 1

    # Final: also try extending to primes 12..15 (exponents 0/1) on top of best
    if time.time() - start_time < DEADLINE:
        # Try adding larger primes to best_n
        best_exps = [0] * NPRIMES
        for i in range(K):
            best_exps[i] = exps_best = 0
        # Reconstruct best_exps from best_n
        m = best_n
        for i, p in enumerate(PRIMES):
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            best_exps[i] = e
        # Try toggling primes 12..15
        for i in range(12, min(NPRIMES, 16)):
            if best_exps[i] == 0 and best_n * PRIMES[i] <= MAX_N:
                cand_n = best_n * PRIMES[i]
                s_c, _ = _tau_perp(cand_n)
                if s_c > best_score:
                    best_score = s_c
                    best_n = cand_n

    return best_n


# Keep a global reference to last best exps for the extension step
exps_best = 0


if __name__ == "__main__":
    n = construct()
    print(n, _tau_perp(n))
