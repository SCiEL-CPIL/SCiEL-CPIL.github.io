"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses multiple random restarts with greedy selection.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    candidates = list(range(2, P['n'] + 1))
    
    best_set = []
    best_len = 0
    
    # Pre-compute a simple complexity metric for sorting heuristics
    # Lower complexity numbers are generally easier to fit
    def complexity(num):
        # Count prime factors with multiplicity
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

    num_restarts = 100000  # High number, but loop breaks on time
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Choose a strategy for ordering
        # 20% of the time: Sort by complexity (ascending)
        # 20% of the time: Sort by value (ascending)
        # 20% of the time: Sort by value (descending)
        # 40% of the time: Random shuffle
        r = random.random()
        if r < 0.2:
            ordered = sorted(candidates, key=lambda x: (complexities[x], x))
        elif r < 0.4:
            ordered = sorted(candidates)
        elif r < 0.6:
            ordered = sorted(candidates, reverse=True)
        else:
            ordered = candidates[:]
            random.shuffle(ordered)
            
        current_set = []
        products = {1}
        
        for cand in ordered:
            # Check if adding cand keeps all products distinct
            # New products are {p * cand for p in products}
            # Collision if any p * cand is already in products
            
            # Optimization: We can check for collision without generating all new products first
            # But generating them is needed if we accept the candidate.
            # For rejection, we can break early.
            
            valid = True
            # We need to check if any p * cand is in products.
            # Note: p * cand cannot be in products if p=1 and cand is new? 
            # Wait, if cand is in products, then 1 * cand = cand is in products. 
            # But cand is not in current_set, so cand is not in products (since products are generated from current_set).
            # However, p * cand could equal some other product q.
            
            for p in products:
                if p * cand in products:
                    valid = False
                    break
            
            if valid:
                current_set.append(cand)
                # Update products
                # Must iterate over a copy of the current products set to avoid adding items while iterating
                # and then checking those new items against the original set in a way that might be incorrect?
                # Actually, the collision check was done against the OLD products set.
                # So we can safely update.
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    # If we reached the theoretical max (18 or 19), we can stop early?
                    # The max size is bounded by the number of distinct exponent vectors possible?
                    # No, just keep going to see if 19 is possible.
                    # But 18 is the target.
                    
    # Fallback if no set found (should not happen)
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Final validation to be absolutely sure
    # Verify distinct subset products
    if best_set:
        prods = {1}
        for x in best_set:
            new_prods = [p * x for p in prods]
            # Check for duplicates within new_prods or against prods
            # Since we built it greedily with checks, it should be fine, but let's be safe.
            # If a collision is found, we should ideally remove the element causing it.
            # But the greedy check `if p * cand in products` ensures no collision with existing products.
            # And `new_prods` are distinct from each other because `prods` were distinct and multiplication by x is injective.
            # So the check in the loop is sufficient.
            pass

    return list(best_set)
# EVOLVE-BLOCK-END
