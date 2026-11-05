"""Rewritten program for maximizing tau_perp(n)."""
import math
import time
from functools import lru_cache
from itertools import product

# EVOLVE-BLOCK-START

def _factorize(n):
    """Factorize n into {prime: exponent}."""
    if n < 2:
        return {}
    factors = {}
    d = 2
    x = n
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def _score_fast(primes, exps):
    """
    Compute tau_perp directly from prime factorization without generating all divisors as integers.
    Uses the structure: divisors are (e1, e2, ..., ek) tuples where 0<=ei<=exps[i].
    Two divisors are coprime iff for every prime, at least one has exponent 0.
    We generate divisors as tuples, sort by their integer value, then count coprime adjacent pairs.
    """
    k = len(primes)
    if k == 0:
        return 0
    
    # Generate all divisor exponent tuples
    ranges = [range(e + 1) for e in exps]
    div_tuples = list(product(*ranges))
    
    # Map each tuple to its integer value for sorting
    def tuple_val(t):
        v = 1
        for i in range(k):
            if t[i] > 0:
                v *= primes[i] ** t[i]
        return v
    
    # Sort by integer value
    div_tuples.sort(key=tuple_val)
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(div_tuples) - 1):
        a = div_tuples[i]
        b = div_tuples[i + 1]
        coprime = True
        for j in range(k):
            if a[j] > 0 and b[j] > 0:
                coprime = False
                break
        if coprime:
            score += 1
    return score


def _score(n):
    """Compute tau_perp(n) from the integer n."""
    if n < 2:
        return 0
    factors = _factorize(n)
    primes = sorted(factors.keys())
    exps = [factors[p] for p in primes]
    return _score_fast(primes, exps)


def _generate_candidates_greedy():
    """
    Greedy construction: build n prime by prime, at each step choosing the exponent
    that maximizes the incremental score contribution, subject to n <= 10^18.
    
    Strategy: try different numbers of primes and different exponent distributions.
    Use a beam search over exponent assignments.
    """
    LIMIT = 10**18
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 2
    best_score = 0
    
    # Precompute prime powers
    prime_powers = {}
    for p in primes_list:
        pp = [1]
        for e in range(1, 60):
            if pp[-1] * p > LIMIT:
                break
            pp.append(pp[-1] * p)
        prime_powers[p] = pp
    
    # Try different numbers of distinct primes
    for num_primes in range(2, 16):
        chosen_primes = primes_list[:num_primes]
        max_exp = [len(prime_powers[p]) - 1 for p in chosen_primes]
        
        # Beam search: maintain top candidates by score
        # State: (current_n, list_of_exponents, current_score_estimate)
        # We'll do a depth-first search with pruning
        
        # For small num_primes, do exhaustive search
        if num_primes <= 4:
            # Exhaustive
            best_local_score = 0
            best_local_exps = None
            for exps in product(*[range(me + 1) for me in max_exp]):
                n = 1
                valid = True
                for i, p in enumerate(chosen_primes):
                    if exps[i] > 0:
                        n *= p ** exps[i]
                    if n > LIMIT:
                        valid = False
                        break
                if not valid:
                    continue
                s = _score_fast(chosen_primes, list(exps))
                if s > best_local_score:
                    best_local_score = s
                    best_local_exps = list(exps)
            
            if best_local_exps and best_local_score > best_score:
                n = 1
                for i, p in enumerate(chosen_primes):
                    n *= p ** best_local_exps[i]
                if n <= LIMIT:
                    best_score = best_local_score
                    best_n = n
        else:
            # For larger num_primes, use greedy + local search
            # Start with all exponents = 1
            exps = [1] * num_primes
            n = 1
            for i, p in enumerate(chosen_primes):
                n *= p
            if n > LIMIT:
                # Can't even fit all with exponent 1
                # Try reducing to fewer primes
                continue
            
            # Local search: try increasing one exponent at a time
            improved = True
            while improved:
                improved = False
                for i in range(num_primes):
                    if exps[i] >= max_exp[i]:
                        continue
                    old_exp = exps[i]
                    old_n = n
                    new_exp = old_exp + 1
                    new_n = n * chosen_primes[i]
                    if new_n > LIMIT:
                        continue
                    exps[i] = new_exp
                    n = new_n
                    s = _score_fast(chosen_primes, exps)
                    if s > best_score:
                        best_score = s
                        best_n = n
                        improved = True
                    else:
                        # Try if this still helps overall (might enable other increases)
                        # For now, revert if it doesn't improve
                        exps[i] = old_exp
                        n = old_n
            
            # Also try a different greedy: start with high exponent on small primes
            for start_exp in range(2, 8):
                exps2 = [0] * num_primes
                exps2[0] = start_exp
                n2 = chosen_primes[0] ** start_exp
                if n2 > LIMIT:
                    continue
                # Fill remaining with 1 if possible
                for i in range(1, num_primes):
                    if n2 * chosen_primes[i] <= LIMIT:
                        exps2[i] = 1
                        n2 *= chosen_primes[i]
                    else:
                        break
                if sum(exps2) >= 2:
                    s2 = _score_fast(chosen_primes, exps2)
                    if s2 > best_score:
                        best_score = s2
                        best_n = n2
    
    return best_n, best_score


def _local_search(n, primes, exps, time_limit=10):
    """Local search: try changing one exponent at a time to improve score."""
    LIMIT = 10**18
    start_time = time.time()
    best_score = _score_fast(primes, exps)
    best_exps = list(exps)
    best_n = n
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(primes)):
            # Try increasing
            old_exp = best_exps[i]
            new_exp = old_exp + 1
            new_n = best_n * primes[i]
            if new_n <= LIMIT:
                best_exps[i] = new_exp
                s = _score_fast(primes, best_exps)
                if s > best_score:
                    best_score = s
                    best_n = new_n
                    improved = True
                else:
                    best_exps[i] = old_exp
            # Try decreasing
            if old_exp > 0:
                new_exp = old_exp - 1
                new_n = best_n // primes[i]
                best_exps[i] = new_exp
                s = _score_fast(primes, best_exps)
                if s > best_score:
                    best_score = s
                    best_n = new_n
                    improved = True
                else:
                    best_exps[i] = old_exp
    
    return best_n, best_score, best_exps


def _try_candidates():
    """Try a set of carefully chosen candidate n values and return the best."""
    LIMIT = 10**18
    primes15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    candidates = []
    
    # Product of first k primes for various k
    for k in range(2, 16):
        n = 1
        for p in primes15[:k]:
            n *= p
            if n > LIMIT:
                break
        if n <= LIMIT and n >= 2:
            candidates.append(n)
    
    # Some structured candidates
    exp_tuples = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [9, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 3, 4, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 5, 4, 3, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 3, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [5, 4, 5, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1],
    ]
    
    for exps in exp_tuples:
        if len(exps) != len(primes15):
            continue
        n = 1
        valid = True
        for i, p in enumerate(primes15):
            if exps[i] > 0:
                n *= p ** exps[i]
            if n > LIMIT:
                valid = False
                break
        if valid and n >= 2:
            candidates.append(n)
    
    # Evaluate all candidates
    best_n = 2
    best_score = 0
    for n in candidates:
        if n < 2 or n > LIMIT:
            continue
        s = _score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n, best_score


def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Phase 1: Try pre-computed candidates
    best_n, best_score = _try_candidates()
    
    # Phase 2: Greedy construction
    greedy_n, greedy_score = _generate_candidates_greedy()
    if greedy_score > best_score:
        best_n = greedy_n
        best_score = greedy_score
    
    # Phase 3: Local search from best candidate
    if time.time() < deadline - 5:
        factors = _factorize(best_n)
        primes = sorted(factors.keys())
        exps = [factors[p] for p in primes]
        ls_n, ls_score, ls_exps = _local_search(best_n, primes, exps, time_limit=10)
        if ls_score > best_score:
            best_n = ls_n
            best_score = ls_score
    
    # Phase 4: If time remains, try more local searches from different starting points
    if time.time() < deadline - 10:
        # Try local search from product of first k primes for various k
        primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for k in range(3, 14):
            if time.time() > deadline - 5:
                break
            n = 1
            for p in primes_list[:k]:
                n *= p
            if n > 10**18:
                break
            primes_k = primes_list[:k]
            exps_k = [1] * k
            ls_n, ls_score, _ = _local_search(n, primes_k, exps_k, time_limit=5)
            if ls_score > best_score:
                best_n = ls_n
                best_score = ls_score
    
    return best_n

# EVOLVE-BLOCK-END
