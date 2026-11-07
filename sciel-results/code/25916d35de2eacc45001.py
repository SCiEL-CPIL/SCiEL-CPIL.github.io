import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

# Precompute prime factorizations for 2..40
_FACTORS = {}

def _prime_factorize(n):
    """Return prime factorization of n as a dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

_FACTORS = {n: _prime_factorize(n) for n in range(2, 41)}

def _can_add_fast(current_products, candidate):
    """
    Check if adding candidate keeps all subset products distinct.
    """
    new_products = {x * candidate for x in current_products}
    return len(new_products & current_products) == 0

def _is_valid_set(lst):
    """Check if a list of integers has all distinct subset products."""
    products = {1}
    for n in lst:
        new_products = {x * n for x in products}
        if new_products & products:
            return False
        products |= new_products
    return True

def _greedy_build(candidates, deadline, rng):
    """
    Greedy build: add candidates in given order if they don't create collisions.
    Returns (result_list, product_set).
    """
    result = []
    current_products = {1}
    for c in candidates:
        if time.time() > deadline:
            break
        if _can_add_fast(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
    return result, current_products

def _local_search(result, current_products, all_candidates, deadline, rng, max_swaps=300):
    """
    Try to improve the set by swapping elements out and in, or adding new elements.
    Returns (best_result, best_products, best_size).
    """
    best_size = len(result)
    best_result = list(result)
    best_products = set(current_products)
    
    # First, try to add any remaining candidates
    for c in all_candidates:
        if time.time() > deadline:
            break
        if c not in result and _can_add_fast(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
            best_size = len(result)
            best_result = list(result)
            best_products = set(current_products)
    
    # Then try swaps - use a more efficient approach
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        if not result:
            break
        
        out_idx = rng.randrange(len(result))
        out_elem = result[out_idx]
        
        # Find candidates not in result
        available = [c for c in all_candidates if c not in result]
        if not available:
            break
        
        in_elem = rng.choice(available)
        
        # Check: remove out_elem, add in_elem
        # Efficient check: temporarily remove out_elem's contribution
        # We need to verify that in_elem doesn't collide with any product
        # that remains after removing out_elem
        
        # For efficiency, just do the full check but break early
        new_set = result[:out_idx] + result[out_idx+1:] + [in_elem]
        
        # Quick check: if in_elem collides with any existing product, skip
        # Build products without out_elem
        products_without = {1}
        for n in new_set:
            if n == out_elem and out_elem in result:
                continue
            new_products = {x * n for x in products_without}
            if new_products & products_without:
                break
            products_without |= new_products
        
        # Now check if in_elem can be added
        if in_elem not in new_set or True:  # in_elem is in new_set
            new_products_in = {x * in_elem for x in products_without}
            if not (new_products_in & products_without):
                # Valid swap
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_result = new_set
                    best_products = products_without | new_products_in
                    result = list(new_set)
                    current_products = set(best_products)
    
    return best_result, best_products, best_size

def _randomized_greedy(all_candidates, deadline, rng, n_restarts=80):
    """
    Multiple randomized greedy restarts to find larger valid sets.
    Uses weighted randomization favoring numbers with fewer prime factors.
    """
    best_result = []
    best_products = {1}
    best_size = 0
    
    # Precompute weights: prefer numbers with fewer distinct prime factors
    weights = []
    for c in all_candidates:
        distinct_primes = len(_FACTORS[c])
        w = 1.0 / (distinct_primes + 0.5)
        weights.append(w)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Weighted random shuffle
        shuffled = list(all_candidates)
        keyed = [(c, rng.random() / (weights[i] + 0.1)) for i, c in enumerate(shuffled)]
        keyed.sort(key=lambda x: x[1])
        shuffled = [k[0] for k in keyed]
        
        result, products = _greedy_build(shuffled, deadline, rng)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
            best_products = products
    
    # Local search on the best result
    if best_size > 0 and time.time() < deadline:
        best_result, best_products, best_size = _local_search(
            best_result, best_products, all_candidates, deadline, rng
        )
    
    return best_result, best_products

def _exhaustive_search_small(candidates, deadline, target_size=18):
    """
    Exhaustive search for a valid set of target_size from candidates.
    Only practical for small candidate sets.
    """
    if len(candidates) < target_size:
        return None
    
    # Use backtracking with pruning
    found = []
    
    def backtrack(start, current, products):
        if time.time() > deadline:
            return False
        if len(current) == target_size:
            found.extend(current)
            return True
        
        remaining = len(candidates) - start
        needed = target_size - len(current)
        if remaining < needed:
            return False
        
        for i in range(start, len(candidates)):
            c = candidates[i]
            if _can_add_fast(products, c):
                current.append(c)
                new_products = products | {x * c for x in products}
                if backtrack(i + 1, current, new_products):
                    return True
                current.pop()
        
        return False
    
    backtrack(0, [], {1})
    return found if found else None

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    t0 = time.time()
    deadline = t0 + 55.0  # 55 second budget (5s margin before 60s kill)
    
    rng = random.Random(42)
    
    all_candidates = list(range(2, 41))
    
    # Phase 1: Fast deterministic greedy with sorted order (quick baseline)
    def count_prime_factors(n):
        return len(_FACTORS[n])
    
    sorted_candidates = sorted(all_candidates, key=count_prime_factors)
    base_result, base_products = _greedy_build(sorted_candidates, deadline, rng)
    
    # Phase 2: Randomized greedy with multiple restarts
    rand_result, rand_products = _randomized_greedy(all_candidates, deadline, rng, n_restarts=120)
    
    # Pick the better one
    if len(rand_result) > len(base_result):
        result = rand_result
        current_products = rand_products
    else:
        result = base_result
        current_products = base_products
    
    # Phase 3: More local search iterations on the best result
    if time.time() < deadline:
        result, current_products, _ = _local_search(
            result, current_products, all_candidates, deadline, rng, max_swaps=400
        )
    
    # Phase 4: If we haven't found 18, try a targeted exhaustive search
    if len(result) < 18 and time.time() < deadline:
        # Try to find a set of size 18 by exhaustive search on a reduced candidate set
        # Keep only candidates that are "compatible" with many others
        # First, compute a compatibility score for each candidate
        compat_scores = {}
        for c in all_candidates:
            count = 0
            for d in all_candidates:
                if c != d and _can_add_fast({1, c}, d):
                    count += 1
            compat_scores[c] = count
        
        # Sort by compatibility and try exhaustive search on top candidates
        top_candidates = sorted(all_candidates, key=lambda x: -compat_scores[x])[:25]
        
        exhaust_result = _exhaustive_search_small(top_candidates, deadline, 18)
        if exhaust_result and len(exhaust_result) > len(result):
            result = exhaust_result
    
    # Final phase: one more local search pass
    if time.time() < deadline:
        # Rebuild products for the current result
        current_products = {1}
        for n in result:
            current_products |= {x * n for x in current_products}
        
        result, current_products, _ = _local_search(
            result, current_products, all_candidates, deadline, rng, max_swaps=200
        )
    
    # Ensure all elements are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
