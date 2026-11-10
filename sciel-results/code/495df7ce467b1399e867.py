"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def _can_add(products, cand):
    """Check if adding cand to a set with given product set keeps all products distinct."""
    for p in products:
        if p * cand in products:
            return False
    return True

def _compute_products(selected):
    """Compute the set of all subset products for the given list."""
    products = {1}
    for x in selected:
        new_prods = [p * x for p in products]
        products.update(new_prods)
    return products

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    candidates = list(range(2, 41))
    n = len(candidates)
    
    # Pre-compute prime factorization vectors for each candidate
    # This allows us to check for "linear dependency" conflicts more efficiently in some cases,
    # but the primary check remains the product set collision check.
    # However, for the branch-and-bound, we can use a bitmask approach.
    
    # 1. Try to find a valid set of size 18 via Branch and Bound (MIS on conflict graph)
    # Conflict graph: Node i, j conflict if adding j to a set containing i (and potentially others) 
    # is not the right definition. The condition is: prod(S) is distinct for all S.
    # This is equivalent to: no subset of the selected numbers multiplies to 1 (impossible for >1) 
    # and no two distinct subsets have the same product.
    # This is a "multiplicative independence" problem.
    
    # Let's use a backtracking search with bitmasks for the set of selected indices.
    # But 2^39 is too large.
    # We can prune: if the current set of products has size 2^k, and we add x, we check collisions.
    
    # Let's stick to the greedy + local search approach but optimize it significantly.
    
    best_set = []
    best_len = 0
    best_products = {1}
    
    # Heuristic orders
    # 1. Primes first
    primes = [i for i in candidates if _is_prime(i)]
    non_primes = [i for i in candidates if not _is_prime(i)]
    
    orders = [
        primes + non_primes,
        non_primes + primes,
        sorted(candidates, key=lambda x: (-len(_prime_factors(x)), x)),
        sorted(candidates, key=lambda x: (len(_prime_factors(x)), x)),
        sorted(candidates, reverse=True),
        sorted(candidates),
    ]
    
    # Add random shuffles
    random.seed(42)
    for _ in range(100):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
            
        current_set = []
        products = {1}
        
        for x in order:
            if _can_add(products, x):
                current_set.append(x)
                new_prods = [p * x for p in products]
                products.update(new_prods)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            best_products = products
            
        if best_len >= 18:
            break
            
        # Local Search: Try to swap elements to improve
        if best_len < 18 and len(current_set) >= best_len:
            # Try to remove one element and add two? Or just swap?
            # Simple swap: remove i, try to add j.
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(current_set)):
                    if time.time() > deadline:
                        break
                    removed = current_set[i]
                    remaining = current_set[:i] + current_set[i+1:]
                    rem_products = _compute_products(remaining)
                    
                    for j in candidates:
                        if j in current_set:
                            continue
                        if _can_add(rem_products, j):
                            # Check if this swap improves size? No, size is same.
                            # But maybe it allows future additions?
                            # Let's just track the best set found so far.
                            # This local search is weak. Let's try a different strategy.
                            pass
                            
    # If we haven't found 18, try a more targeted search for 18.
    # We know that a set of 18 must be very structured.
    # Let's try to construct one explicitly.
    # Based on the analysis in the prompt:
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (10)
    # 2-powers: 2, 4, 16 (3)
    # 3-powers: 3, 9 (2)
    # 5-powers: 25 (1) -- wait, 5 is already in primes list.
    # If we take 5, we can take 25.
    # So:
    # Primes (excluding 2,3,5): 7, 11, 13, 17, 19, 23, 29, 31, 37 (9)
    # 2-powers: 2, 4, 16 (3)
    # 3-powers: 3, 9 (2)
    # 5-powers: 5, 25 (2)
    # Total: 9 + 3 + 2 + 2 = 16.
    # We need 2 more.
    
    # Let's try to add composites to this 16-set.
    base_set = [7, 11, 13, 17, 19, 23, 29, 31, 37, 2, 4, 16, 3, 9, 5, 25]
    base_products = _compute_products(base_set)
    
    for x in candidates:
        if x not in base_set:
            if _can_add(base_products, x):
                base_set.append(x)
                base_products = _compute_products(base_set)
                if len(base_set) >= 18:
                    break
                    
    if len(base_set) > best_len:
        best_len = len(base_set)
        best_set = base_set[:]
        
    if best_len >= 18:
        return [int(x) for x in best_set]
        
    # If still not 18, return the best found.
    if best_set:
        return [int(x) for x in best_set]
        
    # Fallback
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _prime_factors(n):
    """Return dictionary of prime -> exponent for n."""
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

# EVOLVE-BLOCK-END
