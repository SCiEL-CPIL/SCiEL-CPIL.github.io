"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

sys.setrecursionlimit(100000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

# Precompute prime factorizations for 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
def _factorize(n):
    """Return dict of prime -> exponent for n."""
    fac = {}
    for p in _PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1
            n //= p
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

# Precompute factorizations for all candidates 2..40
_FACTORS = {}
for _n in range(2, 41):
    _FACTORS[_n] = _factorize(_n)

def _can_add_fast(products, x):
    """
    Fast check: adding x is valid iff for no p in products does p*x == q for some q in products.
    Equivalently, for no p in products with p*x still in products.
    We check: for each p in products, if p*x in products, collision.
    Optimization: instead of computing p*x (big ints), check divisibility.
    p*x in products  <=>  (p*x) % x == 0 (always true) and (p*x)//x == p, and p*x in products.
    Actually simpler: just check if p*x is in the set. But p*x can be large.
    Alternative: for each q in products, check if q % x == 0 and q//x in products.
    This is O(|products|) with division/mod instead of multiplication.
    """
    for q in products:
        if q % x == 0:
            if q // x in products:
                return False
    return True

def _add_number_fast(products, x):
    """Return new frozenset of products after adding x."""
    new_prods = set(products)
    for p in products:
        new_prods.add(p * x)
    return frozenset(new_prods)

def _greedy_order(order, products):
    """Try adding numbers in given order, return (selected, products)."""
    sel = []
    prods = products
    for a in order:
        if time.time() > _DEADLINE:
            break
        if _can_add_fast(prods, a):
            prods = _add_number_fast(prods, a)
            sel.append(a)
    return sel, prods

def _local_search(selected, products, max_iters=200):
    """
    Hill-climb: try swapping one element for another to increase size or maintain size
    while enabling further additions.
    """
    best_sel = list(selected)
    best_prods = products
    best_size = len(best_sel)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        improved = False
        
        # Try adding any remaining number
        remaining = [a for a in range(2, 41) if a not in best_sel]
        for a in remaining:
            if _can_add_fast(best_prods, a):
                new_prods = _add_number_fast(best_prods, a)
                new_sel = best_sel + [a]
                best_sel = new_sel
                best_prods = new_prods
                best_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add two
        for i, rem in enumerate(best_sel):
            # Remove rem
            # Recompute products without rem
            # This is expensive; skip for now or do it carefully
            pass
    
    return best_sel, best_prods

def construct():
    best_result = []
    best_size = 0
    
    all_candidates = list(range(2, 41))
    
    # Multiple greedy strategies with different orderings
    orders = [
        list(range(2, 41)),                          # natural order
        list(range(40, 1, -1)),                      # reverse
        [p for p in all_candidates if _FACTORS[p] and len(_FACTORS[p]) == 1],  # primes first
        [p for p in all_candidates if p % 2 != 0] + [p for p in all_candidates if p % 2 == 0],  # odds first
    ]
    
    # Add some randomized orders
    rng = random.Random(12345)
    for _ in range(20):
        shuffled = all_candidates[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > _DEADLINE:
            break
        sel, prods = _greedy_order(order, frozenset([1]))
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = sel
            if best_size >= 18:
                return list(best_result)
    
    # Local search / refinement on best solution
    if best_size > 0:
        # Reconstruct products for best_result
        prods = frozenset([1])
        for a in best_result:
            prods = _add_number_fast(prods, a)
        
        # Try to extend by adding any remaining
        for a in range(2, 41):
            if time.time() > _DEADLINE:
                break
            if a not in best_result and _can_add_fast(prods, a):
                prods = _add_number_fast(prods, a)
                best_result.append(a)
                best_size += 1
                if best_size >= 18:
                    return list(best_result)
        
        # Try swaps: remove one element, try to add two
        for i in range(len(best_result)):
            if time.time() > _DEADLINE:
                break
            rem = best_result[i]
            # Recompute products without rem
            # Build products from scratch without rem
            prods2 = frozenset([1])
            for a in best_result:
                if a != rem:
                    prods2 = _add_number_fast(prods2, a)
            
            # Now try to add two numbers
            added = []
            for a in range(2, 41):
                if a == rem or a in best_result:
                    continue
                if _can_add_fast(prods2, a):
                    prods2 = _add_number_fast(prods2, a)
                    added.append(a)
            
            if len(added) >= 1:
                # Net gain if we added more than 1
                new_sel = [a for a in best_result if a != rem] + added
                if len(new_sel) > best_size:
                    best_result = new_sel
                    best_size = len(new_sel)
                    if best_size >= 18:
                        return list(best_result)
    
    # More randomized restarts with local improvement
    rng2 = random.Random(99999)
    for attempt in range(200):
        if time.time() > _DEADLINE:
            break
        
        shuffled = all_candidates[:]
        rng2.shuffle(shuffled)
        
        sel, prods = _greedy_order(shuffled, frozenset([1]))
        
        # Local improvement: try to add more
        for a in range(2, 41):
            if time.time() > _DEADLINE:
                break
            if a not in sel and _can_add_fast(prods, a):
                prods = _add_number_fast(prods, a)
                sel.append(a)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = sel
            if best_size >= 18:
                return list(best_result)
    
    return list(best_result)
# EVOLVE-BLOCK-END
