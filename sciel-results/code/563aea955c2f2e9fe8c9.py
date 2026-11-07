import time

_P0 = time.time()
_DEADLINE = _P0 + 80.0

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """Check if adding x to a set with existing products prods keeps all distinct."""
    for p in prods:
        if p * x in prods:
            return False
    return True

def construct():
    import time
    deadline = _DEADLINE
    
    # Greedy: try to add numbers in various orders
    # Start with primes, then try to add composites
    candidates = list(range(2, 41))
    
    best_set = []
    best_prods = {1}
    
    # Try multiple greedy orders
    import itertools
    
    orders = [
        list(range(2, 41)),
        list(range(40, 1, -1)),
        # Primes first
        [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5)+1))] + 
        [a for a in range(2, 41) if not all(a % d for d in range(2, int(a**0.5)+1))],
        # Composites first
        [a for a in range(2, 41) if not all(a % d for d in range(2, int(a**0.5)+1))] + 
        [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5)+1))],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        cur_set = []
        cur_prods = {1}
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(cur_prods, x):
                cur_set.append(x)
                # Update prods
                new_prods = set()
                for p in cur_prods:
                    new_prods.add(p * x)
                cur_prods = cur_prods | new_prods
        
        if len(cur_set) > len(best_set):
            best_set = cur_set[:]
            best_prods = cur_prods
    
    # Local search: try to swap elements
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                # Remove best_set[i], try to add a new element
                removed = best_set[i]
                # Recompute prods without removed
                # This is expensive, so let's use a different approach
                # Try each candidate not in best_set
                for x in range(2, 41):
                    if x in best_set:
                        continue
                    if time.time() > deadline:
                        break
                    # Check if we can swap: remove best_set[i], add x
                    # We need to check if the set (best_set - {best_set[i]} + {x}) is valid
                    # This requires recomputing subset products, which is expensive
                    # Skip for now, just try adding without removing
            # Try adding elements that weren't added
            for x in range(2, 41):
                if x in best_set:
                    continue
                if _can_add(best_prods, x):
                    best_set.append(x)
                    new_prods = set()
                    for p in best_prods:
                        new_prods.add(p * x)
                    best_prods = best_prods | new_prods
                    improved = True
            if not improved:
                break
    
    return list(best_set)
