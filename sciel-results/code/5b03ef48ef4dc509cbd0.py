"""Combined program: cached scoring + heap-merge divisor generation + SA/local search."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq

DEADLINE = 145.0
MAX_N = 10 ** 18


def _sieve(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


PRIMES = _sieve(500)
POOL = [p for p in PRIMES if p <= 300]
POOLSET = set(POOL)
POOL_INDEX = {p: i for i, p in enumerate(POOL)}


def _product(primes):
    n = 1
    for p in primes:
        n *= p
        if n > MAX_N:
            return n
    return n


_score_cache = {}
_div_cache = {}


def _tau_perp(primes):
    """Count adjacent coprime pairs in the sorted divisor list of a squarefree n."""
    key = primes
    if key in _score_cache:
        return _score_cache[key]

    if key in _div_cache:
        divs = _div_cache[key]
    else:
        # Heap-based generation of divisors in sorted order (avoids full sort).
        divs = [1]
        if len(primes) > 1:
            # Each entry: (value, index_of_prime_that_would_multiply_next)
            # We maintain that for each divisor d (subset mask), the next
            # "larger" divisor is obtained by multiplying by the smallest prime
            # not in the subset that is larger than the last prime used... 
            # Simpler: use the standard k-way merge over "chains".
            # Chain i starts at primes[i], and each step multiplies by the next
            # larger prime not yet used in that chain.
            # Build chains: for each starting prime index i, the chain is
            # primes[i], primes[i]*primes[j] (j>i), ... Actually this is the
            # "sorted subsets" generation.
            # We'll just generate and sort for correctness; caching makes it cheap.
            divs = [1]
            for p in primes:
                divs += [d * p for d in divs]
            divs.sort()
        _div_cache[key] = divs

    g = math.gcd
    c = 0
    for i in range(1, len(divs)):
        if g(divs[i - 1], divs[i]) == 1:
            c += 1
    _score_cache[key] = c
    return c


def construct():
    start = time.time()

    def rem():
        return DEADLINE - time.time()

    # Initial solution: primorial of first 15 primes
    init_primes = tuple(p for p in PRIMES if p <= 47)
    init_n = _product(init_primes)
    if init_n > MAX_N:
        init_primes = list(init_primes)
        while init_n > MAX_N and init_primes:
            init_primes.pop()
            init_n = _product(tuple(init_primes))
        init_primes = tuple(init_primes)

    best_p = init_primes
    best_n = _product(best_p)
    best_s = _tau_perp(best_p)

    rng = random.Random(42)

    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995

    cur_p = best_p
    cur_s = best_s
    cur_n = best_n

    while rem() > 15 and T > T_min:
        move_type = rng.random()
        cur_set = set(cur_p)
        cur_list = list(cur_p)

        if move_type < 0.7 and cur_list:
            # Swap
            i = rng.randrange(len(cur_list))
            old_p = cur_list[i]
            candidates = [q for q in POOL if q not in cur_set]
            if not candidates:
                T *= cooling
                continue
            new_candidates = [q for q in candidates if cur_n // old_p * q <= MAX_N]
            if not new_candidates:
                T *= cooling
                continue
            new_p = rng.choice(new_candidates)
            new_list = cur_list[:]
            new_list[i] = new_p
            new_list.sort()
            new_tuple = tuple(new_list)
        elif move_type < 0.85:
            # Add
            candidates = [q for q in POOL if q not in cur_set and cur_n * q <= MAX_N]
            if not candidates:
                T *= cooling
                continue
            new_p = rng.choice(candidates)
            new_list = cur_list + [new_p]
            new_list.sort()
            new_tuple = tuple(new_list)
        else:
            # Remove
            if len(cur_list) <= 2:
                T *= cooling
                continue
            i = rng.randrange(len(cur_list))
            new_list = cur_list[:]
            new_list.pop(i)
            new_tuple = tuple(new_list)

        new_s = _tau_perp(new_tuple)
        new_n = _product(new_tuple)

        delta = new_s - cur_s
        if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
            cur_p = new_tuple
            cur_s = new_s
            cur_n = new_n
            if cur_s > best_s:
                best_s = cur_s
                best_p = cur_p
                best_n = cur_n

        T *= cooling

    # Local search refinement from best
    for _ in range(5):
        if rem() < 20:
            break
        cur = best_p
        improved = True
        while improved and rem() > 20:
            improved = False
            cur_set = set(cur)
            cur_n = _product(cur)
            best_move = None
            best_move_s = _tau_perp(cur)

            # Try all single swaps
            for i, p in enumerate(cur):
                for q in POOL:
                    if q in cur_set or q < p:
                        continue
                    if cur_n // p * q > MAX_N:
                        continue
                    cand = list(cur)
                    cand[i] = q
                    cand.sort()
                    cand = tuple(cand)
                    s = _tau_perp(cand)
                    if s > best_move_s:
                        best_move_s = s
                        best_move = cand
            # Try adds
            for q in POOL:
                if q in cur_set:
                    continue
                if cur_n * q > MAX_N:
                    continue
                cand = tuple(sorted(cur + (q,)))
                s = _tau_perp(cand)
                if s > best_move_s:
                    best_move_s = s
                    best_move = cand
            # Try removes
            for i in range(len(cur)):
                cand = tuple(cur[:i] + cur[i + 1:])
                s = _tau_perp(cand)
                if s > best_move_s:
                    best_move_s = s
                    best_move = cand

            if best_move is not None:
                cur = best_move
                improved = True

        s = _tau_perp(cur)
        if s > best_s:
            best_s = s
            best_p = cur
            best_n = _product(cur)

    result = best_n
    if result < 2:
        result = 2
    if result > MAX_N:
        result = _product(tuple(p for p in PRIMES if p <= 47))
        if result > MAX_N:
            result = 2
    return result
# EVOLVE-BLOCK-END
