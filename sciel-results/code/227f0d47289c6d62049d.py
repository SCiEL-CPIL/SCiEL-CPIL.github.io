"""Optimize squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

DEADLINE = 145.0
MAX_N = 10**18

def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

PRIMES = _sieve(10000)
LOG_LIMIT = math.log(MAX_N)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def _divisors_sorted(primes):
    divs = [1]
    for p in primes:
        m = len(divs)
        divs += [divs[i] * p for i in range(m)]
    divs.sort()
    return divs

def _tau_perp(divs):
    if len(divs) < 2:
        return 0
    if len(divs) > 1000:
        arr = np.array(divs, dtype=np.int64)
        gcds = np.gcd(arr[:-1], arr[1:])
        return int(np.sum(gcds == 1))
    else:
        c = 0
        prev = divs[0]
        for x in divs[1:]:
            if math.gcd(prev, x) == 1:
                c += 1
            prev = x
        return c

def _product(primes):
    r = 1
    for p in primes:
        r *= p
        if r > MAX_N:
            return r
    return r

def construct():
    start = time.time()
    rng = random.Random(42)

    best_score = -1
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    # Cache for evaluated prime sets to avoid recomputation
    eval_cache = {}

    def eval_set(primes):
        nonlocal best_score, best_n
        # Use sorted tuple as cache key
        key = tuple(sorted(primes))
        if key in eval_cache:
            s = eval_cache[key]
            if s > best_score:
                best_score = s
                best_n = _product(key)
            return s
        
        n = _product(primes)
        if n > MAX_N or n < 2:
            eval_cache[key] = -1
            return -1
        divs = _divisors_sorted(primes)
        s = _tau_perp(divs)
        eval_cache[key] = s
        if s > best_score:
            best_score = s
            best_n = n
        return s

    # Phase 1: Find best k using smallest primes
    best_k = 8
    for k in range(2, 17):
        if time.time() - start > DEADLINE - 100:
            break
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        s = eval_set(cand)
        if s > best_score:
            best_k = k

    # Phase 2: Local search - try to improve by swapping primes
    # Search multiple k values to avoid getting stuck in local optima of a single k
    feasible_ks = []
    for k in range(2, 16):
        if _product(PRIMES[:k]) <= MAX_N:
            feasible_ks.append(k)
        else:
            break
            
    # Sort feasible k by potential (larger k first? or mix?)
    # Let's try all, but prioritize larger k as they have higher upper bound
    # But smaller k might be better? 
    # We'll iterate through them.
    for k in feasible_ks:
        if time.time() - start > DEADLINE - 40:
            break
        if _product(PRIMES[:k]) > MAX_N:
            continue
        
        cur = list(PRIMES[:k])
        cur_set = set(cur)
        cur_n = _product(cur)
        cur_score = eval_set(cur)
        
        # Hill climb: repeatedly find best single swap
        for it in range(300):
            if time.time() - start > DEADLINE - 40:
                break
            best_swap = None
            best_swap_score = cur_score
            
            for i in range(k):
                if time.time() - start > DEADLINE - 40:
                    break
                removed = cur[i]
                # Try swapping with primes in a window around removed's index
                idx = PRIME_TO_IDX.get(removed, 0)
                # Wider window for exploration
                lo = max(0, idx - 20)
                hi = min(len(PRIMES), idx + 200)
                for j in range(lo, hi):
                    np_p = PRIMES[j]
                    if np_p in cur_set:
                        continue
                    new_n = cur_n // removed * np_p
                    if new_n > MAX_N:
                        continue
                    new_list = cur[:i] + [np_p] + cur[i+1:]
                    new_list.sort()
                    s = eval_set(new_list)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap = (i, np_p, new_list, new_n, s)
            
            if best_swap is not None:
                i, np_p, new_list, new_n, s = best_swap
                cur = new_list
                cur_n = new_n
                cur_set = set(cur)
                cur_score = s
                if s > best_score:
                    best_score = s
                    best_n = new_n
            else:
                break

    # Phase 3: Simulated annealing with random restarts
    if time.time() - start < DEADLINE - 30:
        def factorize(n):
            res = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    res.append(d)
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                res.append(n)
            return res
        
        # Start from best_n's factorization
        try:
            seed_primes = factorize(best_n)
        except Exception:
            seed_primes = PRIMES[:8]
        
        k0 = len(seed_primes)
        
        for restart in range(60):
            if time.time() - start > DEADLINE - 25:
                break
            
            # Choose k
            k = k0 + rng.choice([-1, 0, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            
            # Initialize: mix of small primes and some random
            if rng.random() < 0.3:
                # Start from best
                cur = sorted(seed_primes[:k]) if len(seed_primes) >= k else list(PRIMES[:k])
            else:
                # Random start biased toward smaller primes
                idx_pool = list(range(0, 80))
                rng.shuffle(idx_pool)
                chosen = sorted(PRIMES[i] for i in idx_pool[:k])
                if _product(chosen) > MAX_N:
                    chosen = list(PRIMES[:k])
                    if _product(chosen) > MAX_N:
                        continue
                cur = list(chosen)
            
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = eval_set(cur)
            
            # Simulated annealing
            T = 5.0
            T_min = 0.01
            cooling = 0.995
            
            for it in range(300):
                if time.time() - start > DEADLINE - 25:
                    break
                
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIME_TO_IDX.get(removed, 0)
                
                # Candidate new primes
                lo = max(0, idx - 10)
                hi = min(len(PRIMES), idx + 150)
                cands = []
                for j in range(lo, hi):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        continue
                    cands.append(np)
                
                if not cands:
                    continue
                
                np = rng.choice(cands)
                new_list = cur[:i] + [np] + cur[i+1:]
                new_list.sort()
                s = eval_set(new_list)
                new_n = _product(new_list)
                
                delta = s - cur_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                
                T *= cooling
                if T < T_min:
                    T = T_min

    # Phase 4: Final exhaustive polish
    if time.time() - start < DEADLINE - 5:
        def factorize(n):
            res = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    res.append(d)
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                res.append(n)
            return res
        
        try:
            best_primes = factorize(best_n)
            k = len(best_primes)
            cur = sorted(best_primes)
            cur_set = set(cur)
            cur_n = best_n
            cur_score = best_score
            
            # Multiple rounds of best-swap improvement with wider window
            for round_num in range(15):
                if time.time() - start > DEADLINE - 3:
                    break
                best_swap = None
                best_swap_score = cur_score
                for i in range(k):
                    if time.time() - start > DEADLINE - 3:
                        break
                    removed = cur[i]
                    if removed not in PRIME_TO_IDX:
                        continue
                    idx = PRIME_TO_IDX[removed]
                    lo = max(0, idx - 30)
                    hi = min(len(PRIMES), idx + 300)
                    for j in range(lo, hi):
                        np = PRIMES[j]
                        if np in cur_set:
                            continue
                        new_n = cur_n // removed * np
                        if new_n > MAX_N:
                            continue
                        new_list = cur[:i] + [np] + cur[i+1:]
                        new_list.sort()
                        s = eval_set(new_list)
                        if s > best_swap_score:
                            best_swap_score = s
                            best_swap = (i, np, new_list, new_n, s)
                if best_swap is not None:
                    i, np, new_list, new_n, s = best_swap
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    best_score = s
                    best_n = new_n
                else:
                    break
        except Exception:
            pass

    return best_n
# EVOLVE-BLOCK-END
