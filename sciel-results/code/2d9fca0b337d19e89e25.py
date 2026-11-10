import time
import random
from math import isqrt

# Primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _prime_exponent_vector(n):
    """Return tuple of prime exponents for n (length _NUM_PRIMES)."""
    vec = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

def _subset_products_distinct(selected_vecs):
    """
    Check if all 2^k subset products are distinct.
    selected_vecs: list of prime exponent vectors (tuples).
    Two subsets have the same product iff their exponent vectors are equal.
    So we need all subset-sum vectors to be distinct.
    """
    k = len(selected_vecs)
    if k == 0:
        return True
    seen = {tuple([0] * _NUM_PRIMES)}
    # Build all subset sums incrementally
    for vec in selected_vecs:
        new_sums = []
        for s in seen:
            ns = tuple(s[i] + vec[i] for i in range(_NUM_PRIMES))
            if ns in seen:
                return False
            new_sums.append(ns)
        seen.update(new_sums)
    return True

def _can_add_vector(products_vecs, cand_vec):
    """
    Check if adding cand_vec to a set with given product vectors keeps all distinct.
    products_vecs is a set of all subset product exponent vectors.
    """
    for pv in products_vecs:
        nv = tuple(pv[i] + cand_vec[i] for i in range(_NUM_PRIMES))
        if nv in products_vecs:
            return False
    return True

def _add_element_vec(products_vecs, cand_vec):
    """Add element to products set, return new products set."""
    new_vecs = {tuple(pv[i] + cand_vec[i] for i in range(_NUM_PRIMES)) for pv in products_vecs}
    return products_vecs | new_vecs

def _build_conflict_graph(candidates):
    """
    Build conflict graph: edge between i,j if adding both causes a collision.
    Two numbers conflict if their exponent vectors are linearly dependent in a way
    that creates duplicate subset products. Actually, the simplest conflict:
    a and b conflict if a*b equals some product of other pairs... 
    But for pairwise: a and b conflict iff a == b (impossible) or if there exist
    subsets S1, S2 of {a,b} with S1 != S2 but same product.
    For two elements: subsets are {}, {a}, {b}, {a,b}. Products: 1, a, b, ab.
    Collision iff a==1 (no), b==1 (no), a==b (no), a==ab (b=1 no), b==ab (a=1 no).
    So no pairwise conflicts for distinct numbers >= 2!
    
    The conflicts are higher-order. So we can't use a simple pairwise conflict graph.
    Instead, we use the direct product-set approach with branch and bound.
    """
    return None

def _branch_and_bound(candidates, best_size, deadline):
    """
    Branch and bound to find maximum subset with distinct subset products.
    candidates: list of (value, exponent_vector)
    Uses recursive backtracking with pruning.
    """
    n = len(candidates)
    if n == 0:
        return []
    
    best = []
    best_len = best_size
    
    # Order candidates by some heuristic (fewer prime factors first, then value)
    def _score(c):
        v = c[1]
        total = sum(v)
        distinct = sum(1 for x in v if x > 0)
        return (distinct, total, c[0])
    
    ordered = sorted(candidates, key=_score)
    
    # Precompute for each position, which later candidates can be added
    # without immediate pairwise issues (though as noted, no pairwise issues exist)
    
    def _dfs(start, current_set, current_products, remaining):
        nonlocal best, best_len
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best = current_set[:]
            if best_len >= 18:
                return
        
        # Upper bound: len(current_set) + remaining
        if len(current_set) + remaining <= best_len:
            return
        
        # Try adding candidates from start onward
        for i in range(start, n):
            if time.time() > deadline:
                return
            
            val, vec = ordered[i]
            
            if _can_add_vector(current_products, vec):
                new_products = _add_element_vec(current_products, vec)
                current_set.append(val)
                _dfs(i + 1, current_set, new_products, remaining - (n - i))
                current_set.pop()
                
                if best_len >= 18:
                    return
    
    _dfs(0, [], {tuple([0] * _NUM_PRIMES)}, n)
    
    return best

def _greedy_with_pruning(candidates, deadline):
    """
    Greedy construction with smart ordering and backtracking on failure.
    """
    def _score(c):
        v = c[1]
        total = sum(v)
        distinct = sum(1 for x in v if x > 0)
        return (distinct, total, c[0])
    
    ordered = sorted(candidates, key=_score)
    current_set = []
    products = {tuple([0] * _NUM_PRIMES)}
    
    for val, vec in ordered:
        if time.time() > deadline:
            break
        if _can_add_vector(products, vec):
            current_set.append(val)
            products = _add_element_vec(products, vec)
    
    return current_set, products

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin
    best_set = []
    best_len = 0
    
    all_values = list(range(2, 41))
    all_candidates = [(v, _prime_exponent_vector(v)) for v in all_values]
    
    # Precompute prime powers
    def _is_prime_power(n):
        if n < 2:
            return False
        d = 2
        while d * d <= n:
            if n % d == 0:
                m = n
                while m % d == 0:
                    m //= d
                return m == 1
            d += 1
        return True
    
    prime_powers = [v for v in all_values if _is_prime_power(v)]
    
    # Strategy 1: Branch and bound with ordered candidates
    if time.time() < deadline:
        result = _branch_and_bound(all_candidates, best_len, deadline)
        if len(result) > best_len:
            best_len = len(result)
            best_set = result[:]
    
    # Strategy 2: Greedy with multiple orderings
    if best_len < 18 and time.time() < deadline:
        orderings = [
            lambda c: (sum(1 for x in c[1] if x > 0), sum(c[1]), c[0]),
            lambda c: (sum(1 for x in c[1] if x > 0), -c[0]),
            lambda c: (sum(c[1]), c[0]),
            lambda c: c[0],
            lambda c: -c[0],
            lambda c: (not _is_prime_power(c[0]), sum(1 for x in c[1] if x > 0), c[0]),
        ]
        
        for order_key in orderings:
            if time.time() > deadline or best_len >= 18:
                break
            
            ordered = sorted(all_candidates, key=order_key)
            current_set = []
            products = {tuple([0] * _NUM_PRIMES)}
            
            for val, vec in ordered:
                if time.time() > deadline:
                    break
                if _can_add_vector(products, vec):
                    current_set.append(val)
                    products = _add_element_vec(products, vec)
            
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
                if best_len >= 18:
                    break
    
    # Strategy 3: Random restarts with greedy
    if best_len < 18 and time.time() < deadline:
        for restart in range(2000):
            if time.time() > deadline or best_len >= 18:
                break
            
            pool = all_candidates[:]
            random.shuffle(pool)
            
            current_set = []
            products = {tuple([0] * _NUM_PRIMES)}
            
            for val, vec in pool:
                if time.time() > deadline:
                    break
                if _can_add_vector(products, vec):
                    current_set.append(val)
                    products = _add_element_vec(products, vec)
            
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
                if best_len >= 18:
                    break
    
    # Strategy 4: Local search (swap-based)
    if best_len < 18 and time.time() < deadline and best_set:
        for _ in range(100):
            if time.time() > deadline or best_len >= 18:
                break
            
            current_set = best_set[:]
            # Recompute products
            products = {tuple([0] * _NUM_PRIMES)}
            for val in current_set:
                vec = _prime_exponent_vector(val)
                products = _add_element_vec(products, vec)
            
            improved = False
            # Try 1-for-1 swaps
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                rem_products = {tuple([0] * _NUM_PRIMES)}
                for elem in remaining:
                    vec = _prime_exponent_vector(elem)
                    rem_products = _add_element_vec(rem_products, vec)
                
                for val in all_values:
                    if val in current_set:
                        continue
                    vec = _prime_exponent_vector(val)
                    if _can_add_vector(rem_products, vec):
                        new_set = remaining + [val]
                        best_set = new_set
                        best_len = len(new_set)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Final validation
    if best_set:
        vecs = [_prime_exponent_vector(v) for v in best_set]
        if not _subset_products_distinct(vecs):
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return [int(x) for x in best_set]
