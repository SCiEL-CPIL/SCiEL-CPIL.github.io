import time
import random

def _can_add(products, cand):
    """
    Check if adding cand to a set with given product set keeps all products distinct.
    products is a set of all subset products of the current set.
    cand is the new candidate integer.
    """
    # New products are {p * cand for p in products}
    # Collision if any p * cand is already in products
    # Note: p * cand == p' * cand' is impossible if p, p' in products and cand is new,
    # unless p * cand == p'' for some p'' in products.
    # So we just check if p * cand is in products.
    for p in products:
        if p * cand in products:
            return False
    return True

def _local_search(current_set, products, candidates, deadline):
    """
    Try to improve current_set via swap operations.
    current_set: list of integers
    products: set of subset products of current_set
    candidates: list of all possible integers (2..40)
    deadline: time.time() + limit
    """
    k = len(current_set)
    if k >= 18:
        return current_set, products
    
    current_set = list(current_set)
    products = set(products)
    
    # Try swapping elements: remove one from current_set, add one from candidates
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set_copy = list(current_set)
        
        # Shuffle to avoid bias in swap order
        random.shuffle(current_set_copy)
        
        for i in range(len(current_set_copy)):
            if time.time() > deadline:
                break
            # Remove current_set_copy[i]
            removed = current_set_copy[i]
            remaining = [x for x in current_set if x != removed]
            
            # Compute products of remaining
            rem_products = {1}
            for elem in remaining:
                new_prods = [p * elem for p in rem_products]
                rem_products.update(new_prods)
            
            # Try adding each candidate
            # Sort candidates by some heuristic? Maybe just iterate.
            # To be faster, we can skip candidates that are too large?
            # Or just try all.
            for cand in candidates:
                if cand in current_set:
                    continue
                if _can_add(rem_products, cand):
                    # This swap is valid
                    new_set = remaining + [cand]
                    new_products = set(rem_products)
                    new_prods = [p * cand for p in rem_products]
                    new_products.update(new_prods)
                    return new_set, new_products
        # No swap found in this pass
        # If we get here, we are stuck in a local optimum for swaps
        # We could try 2-swaps, but that's expensive.
        # Let's just break out of the while loop if no improvement in a full pass?
        # The 'improved' flag is only set if we return early.
        # So if we finish the loop without returning, improved remains False.
        # So the while loop exits.
        
    return current_set, products

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # Strategy: Random restarts with greedy construction and local search
    # We aim to find a set of size 18.
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        products = {1}
        
        for cand in shuffled:
            if time.time() > deadline:
                break
            
            if _can_add(products, cand):
                current_set.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len >= 18:
                        break
        
        if best_len >= 18:
            break
        
        # Local search: try to improve via swaps
        if len(current_set) >= best_len and len(current_set) < 18:
            improved_set, improved_products = _local_search(
                current_set, products, candidates, deadline
            )
            if len(improved_set) > best_len:
                best_len = len(improved_set)
                best_set = improved_set[:]
                if best_len >= 18:
                    break
    
    # If we didn't find 18, try a few deterministic orderings
    if best_len < 18 and time.time() < deadline:
        # Sort by prime factor count (fewer prime factors = more "independent")
        def prime_factor_count(n):
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
        
        for order_key in [
            lambda x: (prime_factor_count(x), x),
            lambda x: (-prime_factor_count(x), x),
            lambda x: (x % 2, prime_factor_count(x), x),
            lambda x: x,
            lambda x: -x,
        ]:
            if time.time() > deadline or best_len >= 18:
                break
            
            ordered = sorted(candidates, key=order_key)
            current_set = []
            products = {1}
            
            for cand in ordered:
                if time.time() > deadline:
                    break
                if _can_add(products, cand):
                    current_set.append(cand)
                    new_prods = [p * cand for p in products]
                    products.update(new_prods)
                    
                    if len(current_set) > best_len:
                        best_len = len(current_set)
                        best_set = current_set[:]
                        if best_len >= 18:
                            break
            
            if best_len >= 18:
                break
            
            # Local search on this result
            if len(current_set) >= best_len and len(current_set) < 18:
                improved_set, improved_products = _local_search(
                    current_set, products, candidates, deadline
                )
                if len(improved_set) > best_len:
                    best_len = len(improved_set)
                    best_set = improved_set[:]
    
    # Final validation
    if best_set:
        # Verify distinct products
        prods = {1}
        valid = True
        for i in best_set:
            new_prods = [p * i for p in prods]
            if any(p in prods for p in new_prods):
                valid = False
                break
            prods.update(new_prods)
        
        if not valid:
            # Fallback to known valid set (primes)
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return best_set
