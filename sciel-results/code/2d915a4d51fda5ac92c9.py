"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _prime_factor_count(n):
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _greedy_build(candidates_order, deadline=None):
    """Greedy: add numbers in given order if they don't cause collision."""
    selected = []
    products = {1}
    
    for num in candidates_order:
        if deadline is not None and time.time() > deadline:
            break
        new_products = set()
        for p in products:
            new_products.add(p * num)
        if new_products & products:
            continue
        selected.append(num)
        products |= new_products
    return selected, products

def _try_build_random(k_target, n_max=40, deadline=None, rng=None):
    """Try to build a set of k_target distinct integers in 2..n_max with distinct subset products."""
    if rng is None:
        rng = random.Random()
    
    candidates = list(range(2, n_max + 1))
    
    def priority(x):
        if x in [2, 4, 8, 16, 32]:
            return 0
        if x in [3, 9, 27]:
            return 1
        if x in [5, 25]:
            return 2
        if x in [7, 11, 13, 17, 19, 23, 29, 31, 37]:
            return 3
        return 4
    
    for attempt in range(300):
        if deadline is not None and time.time() > deadline:
            break
        
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: (priority(x), rng.random()))
        
        selected = []
        products = {1}
        
        for num in shuffled:
            if len(selected) >= k_target:
                break
            if deadline is not None and time.time() > deadline:
                break
            new_products = set()
            for p in products:
                new_products.add(p * num)
            if new_products & products:
                continue
            selected.append(num)
            products |= new_products
        
        if len(selected) == k_target:
            if _verify_subset_products(selected):
                return selected
    
    return None

def _local_search_improve(initial_set, candidates, deadline=None):
    """Try to improve a set by swapping out elements for better ones."""
    if not initial_set:
        return initial_set
    
    current = list(initial_set)
    selected_set = set(current)
    
    # Build current products
    products = {1}
    for n in current:
        new_products = set()
        for p in products:
            new_products.add(p * n)
        products |= new_products
    
    # Try adding each non-selected number
    for n in sorted(candidates, key=lambda x: _prime_factor_count(x)):
        if n in selected_set:
            continue
        if deadline is not None and time.time() > deadline:
            break
        new_products = set()
        for p in products:
            new_products.add(p * n)
        if not (new_products & products):
            current.append(n)
            selected_set.add(n)
            products |= new_products
    
    # Try swapping: remove one element, try to add a better one
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(current)):
            if deadline is not None and time.time() > deadline:
                break
            # Remove current[i]
            removed = current[i]
            current_without = current[:i] + current[i+1:]
            sel_without = selected_set - {removed}
            
            # Rebuild products without removed
            prods_without = {1}
            for n in current_without:
                new_p = set()
                for p in prods_without:
                    new_p.add(p * n)
                prods_without |= new_p
            
            # Try adding each non-selected number
            best_add = None
            best_prods = None
            for n in sorted(candidates, key=lambda x: _prime_factor_count(x)):
                if n in sel_without or n == removed:
                    continue
                new_products = set()
                for p in prods_without:
                    new_products.add(p * n)
                if not (new_products & prods_without):
                    # Check if this is "better" - for now just take first valid
                    best_add = n
                    best_prods = prods_without | new_products
                    break
            
            if best_add is not None:
                # Perform the swap
                current = current_without + [best_add]
                selected_set = sel_without | {best_add}
                products = best_prods
                improved = True
                break
    
    return current

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin
    
    candidates = list(range(2, 41))
    
    # Known good set of 16
    known_16 = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best = None
    if _verify_subset_products(known_16):
        best = known_16
    
    # Try to find 17
    if best is None or len(best) < 17:
        # First try local search from known 16
        improved = _local_search_improve(known_16, candidates, deadline=deadline)
        if len(improved) > (len(best) if best else 0):
            best = improved
            if _verify_subset_products(best):
                pass
            else:
                best = known_16
        
        # Try random search for 17
        if best is None or len(best) < 17:
            rng = random.Random(42)
            result = _try_build_random(17, deadline=deadline, rng=rng)
            if result:
                best = result
    
    # Try to find 18
    if best is None or len(best) < 18:
        rng = random.Random(123)
        result = _try_build_random(18, deadline=deadline, rng=rng)
        if result:
            best = result
        
        # Also try local search from best so far
        if best:
            improved = _local_search_improve(best, candidates, deadline=deadline)
            if len(improved) > len(best) and _verify_subset_products(improved):
                best = improved
    
    # Final verification
    if best is None or not _verify_subset_products(best):
        # Fallback to primes
        from math import isqrt
        best = [a for a in range(2, 41) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Ensure all are Python ints in range 2..40
    result = []
    for x in best:
        if isinstance(x, bool):
            continue
        if not isinstance(x, int):
            continue
        if 2 <= x <= 40:
            result.append(x)
    
    # Ensure distinct
    seen = set()
    unique = []
    for x in result:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    
    return unique

# EVOLVE-BLOCK-END
