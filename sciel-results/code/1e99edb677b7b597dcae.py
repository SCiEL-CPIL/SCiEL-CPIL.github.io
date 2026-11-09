"""Maximize coprime adjacent divisor pairs for n <= 1e18 (simulated annealing on prime sets)."""
# EVOLVE-BLOCK-START
import time
import math
import random

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
# Larger pool: primes up to 300 for more diverse search
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


_tau_perp_cache = {}

def _tau_perp(primes):
    """Count adjacent coprime pairs in the sorted divisor list of a squarefree n."""
    if not primes:
        return 0
    key = primes
    if key in _tau_perp_cache:
        return _tau_perp_cache[key]
    divs = [1]
    for p in primes:
        current_len = len(divs)
        for i in range(current_len):
            divs.append(divs[i] * p)
    divs.sort()
    g = math.gcd
    c = 0
    for i in range(1, len(divs)):
        if g(divs[i - 1], divs[i]) == 1:
            c += 1
    _tau_perp_cache[key] = c
    return c


def construct():
    start = time.time()

    def rem():
        return DEADLINE - time.time()

    # Start with primorial of first 15 primes as initial solution
    init_primes = tuple(p for p in PRIMES if p <= 47)  # first 15 primes
    init_n = _product(init_primes)
    if init_n > MAX_N:
        # Remove largest until fits
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

    while rem() > 5 and T > T_min:
        if len(_tau_perp_cache) > 50000:
            _tau_perp_cache.clear()
        # Generate a move: swap one prime, add one, or remove one
        move_type = rng.random()
        cur_set = set(cur_p)
        cur_list = list(cur_p)
        
        new_n = cur_n
        if move_type < 0.7 and cur_list:
            # Swap: replace one prime with another
            i = rng.randrange(len(cur_list))
            old_p = cur_list[i]
            # Choose a new prime from pool, biased toward similar magnitude
            # to keep product in range
            candidates = [q for q in POOL if q not in cur_set]
            if len(candidates) > 100:
                candidates = rng.sample(candidates, 100)
            if not candidates:
                T *= cooling
                continue
            # Filter by product constraint
            new_candidates = []
            for q in candidates:
                new_prod = cur_n // old_p * q
                if new_prod <= MAX_N:
                    new_candidates.append(q)
            if not new_candidates:
                T *= cooling
                continue
            new_p = rng.choice(new_candidates)
            new_n = cur_n // old_p * new_p
            new_list = cur_list[:]
            new_list[i] = new_p
            new_list.sort()
            new_tuple = tuple(new_list)
        elif move_type < 0.85:
            # Add a prime
            candidates = [q for q in POOL if q not in cur_set and cur_n * q <= MAX_N]
            if len(candidates) > 50:
                candidates = rng.sample(candidates, 50)
            if not candidates:
                T *= cooling
                continue
            new_p = rng.choice(candidates)
            new_n = cur_n * new_p
            new_list = cur_list + [new_p]
            new_list.sort()
            new_tuple = tuple(new_list)
        else:
            # Remove a prime
            if len(cur_list) <= 2:
                T *= cooling
                continue
            i = rng.randrange(len(cur_list))
            old_p = cur_list[i]
            new_n = cur_n // old_p
            new_list = cur_list[:]
            new_list.pop(i)
            new_tuple = tuple(new_list)

        new_s = _tau_perp(new_tuple)
        
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

    # Fast local search refinement from best
    # Only try a subset of moves to keep it fast
    if rem() > 5:
        _tau_perp_cache.clear()
        cur = best_p
        cur_n = best_n
        cur_s = best_s
        cur_set = set(cur)
        
        # Try a few random restarts of local search or just a few iterations
        for _ in range(3):
            if rem() < 5:
                break
            improved = True
            while improved and rem() > 5:
                improved = False
                best_move = None
                best_move_s = cur_s
                
                # Try swaps with a limited pool (e.g., primes near the current ones)
                # Or just random samples from POOL if POOL is large
                # For simplicity, we'll try all swaps but with a time check
                # To make it faster, we'll limit the candidates to 50 random choices per prime
                for i, p in enumerate(cur):
                    # Get a sample of candidates
                    candidates = [q for q in POOL if q not in cur_set]
                    if not candidates:
                        continue
                    # Randomly sample up to 20 candidates to keep it fast
                    sample_size = min(20, len(candidates))
                    sampled = rng.sample(candidates, sample_size)
                    for q in sampled:
                        new_n = cur_n // p * q
                        if new_n > MAX_N:
                            continue
                        cand = list(cur)
                        cand[i] = q
                        cand.sort()
                        cand = tuple(cand)
                        s = _tau_perp(cand)
                        if s > best_move_s:
                            best_move_s = s
                            best_move = cand
                
                # Try adds (limited)
                candidates_add = [q for q in POOL if q not in cur_set and cur_n * q <= MAX_N]
                if candidates_add:
                    sample_size = min(20, len(candidates_add))
                    sampled = rng.sample(candidates_add, sample_size)
                    for q in sampled:
                        cand = tuple(sorted(cur + (q,)))
                        s = _tau_perp(cand)
                        if s > best_move_s:
                            best_move_s = s
                            best_move = cand

                # Try removes (all, since there are few)
                for i in range(len(cur)):
                    cand = tuple(cur[:i] + cur[i+1:])
                    s = _tau_perp(cand)
                    if s > best_move_s:
                        best_move_s = s
                        best_move = cand
                
                if best_move is not None:
                    cur = best_move
                    cur_s = best_move_s
                    cur_n = _product(cur)
                    cur_set = set(cur)
                    improved = True
                    if cur_s > best_s:
                        best_s = cur_s
                        best_p = cur
                        best_n = cur_n

    return best_n
# EVOLVE-BLOCK-END
