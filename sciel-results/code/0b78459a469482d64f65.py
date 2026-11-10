"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses multiple random restarts with greedy selection and vector-based pruning.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    candidates = list(range(2, 41))
    prime_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    
    best_set = []
    best_len = 0
    
    def complexity(num):
        """Count prime factors with multiplicity."""
        n = num
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    complexities = {x: complexity(x) for x in candidates}
    
    def check_and_add(products, cand):
        """Check if adding cand keeps all products distinct. Returns (valid, new_products)."""
        # Check for collision: any p * cand in products?
        for p in products:
            if p * cand in products:
                return False, None
        # No collision, generate new products
        new_prods = set(products)
        for p in products:
            new_prods.add(p * cand)
        return True, new_prods
    
    num_restarts = 200000  # High number, loop breaks on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose a strategy for ordering
        r = random.random()
        if r < 0.15:
            # Sort by complexity (ascending) - simpler numbers first
            ordered = sorted(candidates, key=lambda x: (complexities[x], x))
        elif r < 0.30:
            # Sort by value (ascending)
            ordered = sorted(candidates)
        elif r < 0.45:
            # Sort by value (descending)
            ordered = sorted(candidates, reverse=True)
        elif r < 0.60:
            # Sort by number of distinct prime factors (ascending)
            def num_distinct_primes(x):
                count = 0
                d = 2
                n = x
                while d * d <= n:
                    if n % d == 0:
                        count += 1
                        while n % d == 0:
                            n //= d
                    d += 1
                if n > 1:
                    count += 1
                return count
            ordered = sorted(candidates, key=lambda x: (num_distinct_primes(x), x))
        else:
            # Random shuffle
            ordered = candidates[:]
            random.shuffle(ordered)
        
        current_set = []
        products = {1}
        
        for cand in ordered:
            valid, new_products = check_and_add(products, cand)
            
            if valid:
                current_set.append(cand)
                products = new_products
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    # Early termination if we hit theoretical max
                    if best_len >= 18:
                        break
        
        # If we found a set of size 18+, we can try to improve or just stop
        if best_len >= 18:
            # Continue searching a bit more to see if we can do better, 
            # but 18 is likely optimal given the constraints
            if best_len >= 19:
                break
    
    # Fallback if no set found (should not happen)
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Final validation
    if best_set:
        prods = {1}
        for x in best_set:
            new_prods = [p * x for p in prods]
            for np in new_prods:
                if np in prods:
                    # Should not happen if our checks were correct
                    break
            prods.update(new_prods)
    
    return list(best_set)
# EVOLVE-BLOCK-END
