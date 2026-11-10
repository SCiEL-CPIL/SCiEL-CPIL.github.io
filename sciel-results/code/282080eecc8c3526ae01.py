import time
import random
from itertools import product

def _check_distinct_products(selected):
    """Check if all 2^k subset products are distinct."""
    if not selected:
        return True
    k = len(selected)
    if k > 18:
        return False
    products = set()
    # Enumerate all subsets via bit masks
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _can_add(products, cand):
    """Check if adding cand to a set with given product set keeps all products distinct."""
    # New products are {p * cand for p in products}
    # Collision if any p * cand is already in products
    for p in products:
        if p * cand in products:
            return False
    return True

def _local_search(current_set, products, candidates, deadline):
    """Try to improve current_set via swap operations."""
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
        for i in range(len(current_set_copy)):
            if time.time() > deadline:
                break
            # Remove current_set_copy[i]
            removed = current_set_copy[i]
            remaining = current_set_copy[:i] + current_set_copy[i+1:]
            
            # Compute products of remaining
            rem_products = {1}
            for elem in remaining:
                new_prods = [p * elem for p in rem_products]
                rem_products.update(new_prods)
            
            # Try adding each candidate
            for cand in candidates:
                if cand in current_set_copy:
                    continue
                if _can_add(rem_products, cand):
                    # This swap is valid
                    new_set = remaining + [cand]
                    new_products = set(rem_products)
                    new_prods = [p * cand for p in rem_products]
                    new_products.update(new_prods)
                    return new_set, new_products
        # No swap found in this pass
    return current_set, products

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin (90s total, leave 35s buffer)
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # Strategy 1: Greedy with random restarts + local search
    num_restarts = 10000
    
    for _ in range(num_restarts):
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
    
    # Strategy 2: Try a few deterministic orderings if time permits
    if best_len < 18 and time.time() < deadline:
        # Try sorting by prime factor count (fewer prime factors = more "independent")
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
    if best_set and not _check_distinct_products(best_set):
        # Fallback to known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return best_set
