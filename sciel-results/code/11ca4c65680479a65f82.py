"""Find the largest subset of {2..40} with distinct subset products."""

import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    candidates = list(range(2, 41))
    
    def get_products(s):
        prods = {1}
        for x in s:
            new_prods = {p * x for p in prods}
            prods.update(new_prods)
        return prods

    def is_valid_set(s):
        if not s:
            return True, {1}
        prods = get_products(s)
        return len(prods) == (1 << len(s)), prods

    best_set = []
    best_size = 0
    best_prods = {1}
    
    # Heuristic 1: Primes first
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    order1 = primes + composites
    
    # Heuristic 2: Sorted by value
    order2 = candidates
    
    # Heuristic 3: Reverse sorted
    order3 = sorted(candidates, reverse=True)
    
    # Heuristic 4: Primes, then composites sorted by value
    order4 = primes + sorted(composites)
    
    for order in [order1, order2, order3, order4]:
        if time.time() > deadline:
            break
            
        current_set = []
        current_prods = {1}
        
        for c in order:
            if time.time() > deadline:
                break
            new_prods = {p * c for p in current_prods}
            if not (new_prods & current_prods):
                current_set.append(c)
                current_prods.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            best_prods = current_prods

    # Local Search on best_set
    if best_size < 18 and time.time() < deadline:
        current_set = best_set[:]
        current_prods = best_prods
        used = set(current_set)
        available = [c for c in candidates if c not in used]
        
        # Limit swaps to prevent infinite loops
        max_swaps = 50
        swaps_done = 0
        
        while swaps_done < max_swaps and time.time() < deadline:
            improved = False
            
            # Try adding
            for c in available:
                new_prods = {p * c for p in current_prods}
                if not (new_prods & current_prods):
                    current_set.append(c)
                    current_prods.update(new_prods)
                    used.add(c)
                    available.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swapping
            swap_made = False
            for i, x in enumerate(current_set):
                if time.time() > deadline:
                    break
                for y in available:
                    if time.time() > deadline:
                        break
                    temp_set = current_set[:]
                    temp_set[i] = y
                    valid, temp_prods = is_valid_set(temp_set)
                    if valid:
                        current_set[i] = y
                        current_prods = temp_prods
                        used.remove(x)
                        used.add(y)
                        available.remove(y)
                        available.append(x)
                        swap_made = True
                        swaps_done += 1
                        improved = True
                        break
                if improved:
                    break
                    
            if not improved:
                break

        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
            best_prods = current_prods

    # Final check
    valid, _ = is_valid_set(best_set)
    if not valid:
        best_set = [p for p in candidates if _is_prime(p)]
        
    return list(best_set)
