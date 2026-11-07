"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _subset_products_valid(elements):
    """Check if all 2^k subset products of elements are distinct."""
    k = len(elements)
    products = set()
    # Use iterative construction: for each element, double the set of products
    prods = [1]
    for e in elements:
        new_prods = [p * e for p in prods]
        prods = prods + new_prods
    if len(prods) != len(set(prods)):
        return False
    return True

def _can_add(elements, candidate, existing_products):
    """Check if adding candidate to elements keeps all subset products distinct.
    existing_products is the set of all subset products of elements."""
    # New subset products are existing_products and existing_products * candidate
    new_prods = existing_products | {p * candidate for p in existing_products}
    return len(new_prods) == 2 * len(existing_products)

def construct():
    start_time = time.time()
    
    candidates = list(range(2, 41))
    
    # Greedy construction with multiple orderings
    best_set = []
    best_size = 0
    
    def time_up():
        return time.time() - start_time > DEADLINE
    
    # Try different orderings of candidates
    orderings = [
        candidates,           # natural order
        sorted(candidates, reverse=True),  # descending
        # Primes first, then composites
        [n for n in candidates if all(n % d for d in range(2, int(n**0.5)+1))] + 
        [n for n in candidates if not all(n % d for d in range(2, int(n**0.5)+1))],
    ]
    
    for ordering in orderings:
        if time_up():
            break
        
        current = []
        current_prods = {1}
        
        for cand in ordering:
            if time_up():
                break
            if _can_add(current, cand, current_prods):
                # Add candidate
                current.append(cand)
                current_prods = current_prods | {p * cand for p in current_prods}
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        # If we hit the theoretical max, stop
        if best_size >= 18:
            break
    
    # Local search: try to improve by swapping
    if best_size < 18 and not time_up():
        # Try to add elements not in best_set by removing some
        in_set = set(best_set)
        not_in_set = [x for x in candidates if x not in in_set]
        
        # Try single additions with removals
        for remove in best_set:
            if time_up():
                break
            temp_set = [x for x in best_set if x != remove]
            # Compute products for temp_set
            prods = {1}
            for e in temp_set:
                prods = prods | {p * e for p in prods}
            
            for add in not_in_set:
                if time_up():
                    break
                if add not in in_set:
                    if _can_add(temp_set, add, prods):
                        new_set = temp_set + [add]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
    
    # Final local search: try all pairs of swaps
    if best_size < 18 and not time_up():
        in_set = set(best_set)
        not_in_set = [x for x in candidates if x not in in_set]
        
        for i, remove in enumerate(best_set):
            if time_up():
                break
            temp_set = [x for x in best_set if x != remove]
            prods = {1}
            for e in temp_set:
                prods = prods | {p * e for p in prods}
            
            for add in not_in_set:
                if time_up():
                    break
                if add not in in_set:
                    if _can_add(temp_set, add, prods):
                        new_set = temp_set + [add]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
    
    # Return as Python ints in a list
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
