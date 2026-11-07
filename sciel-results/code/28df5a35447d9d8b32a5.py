"""Combined program: fast incremental check + deterministic greedy + many randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from collections import Counter

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _prime_factors(a):
    """Return dict of prime -> exponent for integer a."""
    factors = {}
    d = 2
    while d * d <= a:
        while a % d == 0:
            factors[d] = factors.get(d, 0) + 1
            a //= d
        d += 1 if d == 2 else 2
    if a > 1:
        factors[a] = factors.get(a, 0) + 1
    return factors

def _largest_prime_factor(n):
    lpf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            lpf = d
            n //= d
        d += 1
    if n > 1:
        lpf = n
    return lpf

def _check_valid(S):
    """
    Validate that all subset products of S are distinct.
    Uses the incremental set expansion method.
    """
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _try_greedy(order, initial_set=None, initial_prods=None):
    """
    Greedily add elements from order to build a valid set.
    If initial_set and initial_prods are provided, start from there.
    Returns the list of elements added (including initial if provided).
    """
    if initial_set is not None:
        current_set = list(initial_set)
        prods = set(initial_prods)
    else:
        current_set = []
        prods = {1}
    for v in order:
        if v in current_set:
            continue
        new_prods = []
        valid = True
        for p in prods:
            np = v * p
            if np in prods:
                valid = False
                break
            new_prods.append(np)
        if valid:
            current_set.append(v)
            prods.update(new_prods)
    return current_set

def _compute_prods(S):
    """Compute all subset products of S."""
    prods = {1}
    for x in S:
        prods = prods | {p * x for p in prods}
    return prods

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    # Pre-compute prime factorization for each candidate
    factor_cache = {a: _prime_factors(a) for a in candidates}
    
    # Score each candidate by "novelty" - how many new prime factors it brings
    # and how "small" its exponents are
    def _novelty_score(a):
        f = factor_cache[a]
        score = 0
        for p, e in f.items():
            # Primes that are not yet "used" are more valuable
            # We'll compute this dynamically during search
            score += 1 / e  # Lower exponents are better
        return score
    
    # Initial best: primes
    best_set = primes[:]
    best_size = len(best_set)
    best_prods = _compute_prods(best_set)
    
    # 1. Deterministic Greedy Attempts
    # Order 1: Primes + Composites sorted by largest prime factor (descending)
    composites_lpf_desc = sorted(composites, key=_largest_prime_factor, reverse=True)
    order1 = primes + composites_lpf_desc
    sol1 = _try_greedy(order1)
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_set = sol1[:]
        best_prods = _compute_prods(best_set)
        
    # Order 2: Primes + Composites sorted by value (ascending)
    order2 = primes + sorted(composites)
    sol2 = _try_greedy(order2)
    if len(sol2) > best_size:
        best_size = len(sol2)
        best_set = sol2[:]
        best_prods = _compute_prods(best_set)
        
    # Order 3: Primes + Composites sorted by value (descending)
    order3 = primes + sorted(composites, reverse=True)
    sol3 = _try_greedy(order3)
    if len(sol3) > best_size:
        best_size = len(sol3)
        best_set = sol3[:]
        best_prods = _compute_prods(best_set)
    
    # Order 4: Sort by "novelty" - prefer numbers with rare prime factors
    # First, count how often each prime appears in composites
    prime_freq = Counter()
    for c in composites:
        for p in factor_cache[c]:
            prime_freq[p] += 1
    
    def _rarity_score(a):
        f = factor_cache[a]
        score = 0
        for p, e in f.items():
            # Rarer primes are more valuable
            score += 1 / (prime_freq.get(p, 1) * e)
        return -score  # Negate for sorting (higher rarity = lower score value)
    
    composites_rarity = sorted(composites, key=_rarity_score)
    order4 = primes + composites_rarity
    sol4 = _try_greedy(order4)
    if len(sol4) > best_size:
        best_size = len(sol4)
        best_set = sol4[:]
        best_prods = _compute_prods(best_set)

    # 2. Randomized Restart Greedy with Local Search
    rng = random.Random(42)
    
    def _local_search(current_set, current_prods, max_iters=50):
        """
        Try to improve the set by:
        1. Adding any remaining candidate that fits
        2. Swapping one element for two (if it increases size)
        3. Replacing one element with another that allows more additions
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try to add any candidate
            for v in candidates:
                if v in current_set:
                    continue
                new_prods = {p * v for p in current_prods}
                if not (current_prods & new_prods):
                    # Can add v
                    current_set.append(v)
                    current_prods |= new_prods
                    improved = True
                    break
            
            # If we couldn't add, try swaps: remove one, add two
            if not improved:
                for i, v_remove in enumerate(current_set):
                    # Remove v_remove
                    temp_set = current_set[:i] + current_set[i+1:]
                    temp_prods = _compute_prods(temp_set)
                    
                    # Try to add two candidates
                    added = []
                    for v1 in candidates:
                        if v1 in current_set:
                            continue
                        new1 = {p * v1 for p in temp_prods}
                        if not (temp_prods & new1):
                            added.append(v1)
                            temp_prods |= new1
                            break
                    if added:
                        for v2 in candidates:
                            if v2 in current_set or v2 in added:
                                continue
                            new2 = {p * v2 for p in temp_prods}
                            if not (temp_prods & new2):
                                # We can add two, net gain of 1
                                current_set = temp_set + added + [v2]
                                current_prods = _compute_prods(current_set)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return current_set, current_prods
    
    # Run randomized restarts with local search
    while time.time() < deadline:
        # Generate a random order with bias
        if rng.random() < 0.3:
            # Strategy A: Shuffle all
            order = candidates[:]
            rng.shuffle(order)
        elif rng.random() < 0.5:
            # Strategy B: Primes first, then shuffle composites
            p_shuffled = primes[:]
            rng.shuffle(p_shuffled)
            c_shuffled = composites[:]
            rng.shuffle(c_shuffled)
            order = p_shuffled + c_shuffled
        else:
            # Strategy C: Sort by rarity, then shuffle composites within same rarity
            c_sorted = sorted(composites, key=_rarity_score)
            # Group by approximate rarity score
            groups = []
            current_group = []
            prev_score = None
            for c in c_sorted:
                s = _rarity_score(c)
                if prev_score is not None and abs(s - prev_score) > 0.1:
                    if current_group:
                        groups.append(current_group)
                    current_group = [c]
                else:
                    current_group.append(c)
                prev_score = s
            if current_group:
                groups.append(current_group)
            
            # Shuffle within each group
            c_final = []
            for g in groups:
                g_shuffled = g[:]
                rng.shuffle(g_shuffled)
                c_final.extend(g_shuffled)
            
            order = primes + c_final
        
        sol = _try_greedy(order)
        
        # Apply local search to improve
        if len(sol) >= best_size - 1:  # Only do local search if close to best
            sol, sol_prods = _local_search(sol, _compute_prods(sol))
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            best_prods = sol_prods if 'sol_prods' in locals() else _compute_prods(best_set)
        
        # Early termination if we hit a very good solution
        if best_size >= 16:
            # Try a few more iterations but don't spend too much time
            if rng.random() < 0.1:
                break
    
    # Final validation
    if not _check_valid(best_set):
        # Fallback to primes if something went wrong (shouldn't happen)
        best_set = primes[:]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
