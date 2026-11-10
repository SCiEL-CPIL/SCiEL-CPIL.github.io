import time
import random
from itertools import combinations

def _check_distinct_products(selected):
    """Check if all 2^k subset products are distinct."""
    if not selected:
        return True
    k = len(selected)
    if k > 18:
        return False
    products = set()
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
    for p in products:
        if p * cand in products:
            return False
    return True

def _add_element(products, cand):
    """Add element cand to product set, return new product set."""
    new_prods = frozenset(products)
    for p in products:
        new_prods.add(p * cand)
    return new_prods

def _remove_element_and_rebuild(selected, idx):
    """Remove element at idx from selected, rebuild product set."""
    remaining = selected[:idx] + selected[idx+1:]
    products = {1}
    for elem in remaining:
        new_prods = [p * elem for p in products]
        products.update(new_prods)
    return remaining, products

def _local_search(current_set, products, candidates, deadline, max_swaps=5000):
    """Try to improve current_set via swap operations (remove one, add one)."""
    k = len(current_set)
    if k >= 18:
        return current_set, products
    
    current_set = list(current_set)
    products = set(products)
    swaps_done = 0
    
    while swaps_done < max_swaps and time.time() < deadline:
        found_swap = False
        current_set_copy = list(current_set)
        
        for i in range(len(current_set_copy)):
            if time.time() > deadline:
                break
            if found_swap:
                break
            
            removed = current_set_copy[i]
            remaining, rem_products = _remove_element_and_rebuild(current_set, i)
            
            # Try adding each candidate not in remaining
            for cand in candidates:
                if cand in remaining:
                    continue
                if _can_add(rem_products, cand):
                    # Valid swap found
                    new_set = remaining + [cand]
                    new_products = set(rem_products)
                    new_prods = [p * cand for p in rem_products]
                    new_products.update(new_prods)
                    current_set = new_set
                    products = new_products
                    found_swap = True
                    swaps_done += 1
                    break
        
        if not found_swap:
            break
    
    return current_set, products

def _try_backtrack_add(selected, products, candidates, deadline, depth=1):
    """Try to add one more element via backtracking search."""
    if time.time() > deadline:
        return None, None
    
    k = len(selected)
    if k >= 18:
        return selected, products
    
    # Try each candidate
    for cand in candidates:
        if cand in selected:
            continue
        if _can_add(products, cand):
            new_set = selected + [cand]
            new_products = _add_element(products, cand)
            return new_set, new_products
    
    return None, None

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 50  # Safety margin (90s total, leave 40s buffer)
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # Precompute some properties
    def smallest_prime_factor(n):
        for d in range(2, n + 1):
            if n % d == 0:
                return d
        return n
    
    def num_prime_factors(n):
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
    
    spf = {n: smallest_prime_factor(n) for n in candidates}
    npf = {n: num_prime_factors(n) for n in candidates}
    
    # Strategy 1: Multiple greedy orderings with local search
    orderings = []
    
    # Random orderings
    for _ in range(3):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Sorted by various heuristics
    orderings.append(sorted(candidates, key=lambda x: -x))  # descending
    orderings.append(sorted(candidates, key=lambda x: x))   # ascending
    orderings.append(sorted(candidates, key=lambda x: (-spf[x], x)))  # large SPF first
    orderings.append(sorted(candidates, key=lambda x: (npf[x], x)))   # few prime factors first
    orderings.append(sorted(candidates, key=lambda x: (-npf[x], x)))  # many prime factors first
    orderings.append(sorted(candidates, key=lambda x: (x % 2, -x)))   # odds first, then descending
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
        
        current_set = []
        products = {1}
        
        for cand in ordering:
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
        
        # Local search to try to improve
        if len(current_set) < 18 and time.time() < deadline:
            improved_set, improved_products = _local_search(
                current_set, products, candidates, deadline
            )
            if len(improved_set) > best_len:
                best_len = len(improved_set)
                best_set = improved_set[:]
                
                if best_len >= 18:
                    break
    
    # Strategy 2: Random restarts with greedy + local search
    num_restarts = 2000
    for r in range(num_restarts):
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
        
        # Local search
        if len(current_set) < 18 and time.time() < deadline:
            improved_set, improved_products = _local_search(
                current_set, products, candidates, deadline
            )
            if len(improved_set) > best_len:
                best_len = len(improved_set)
                best_set = improved_set[:]
                
                if best_len >= 18:
                    break
    
    # Strategy 3: Try to push to 18 with focused search
    if best_len < 18 and time.time() < deadline:
        # Use the best set found so far as a starting point
        start_set = best_set[:]
        start_products = {1}
        for elem in start_set:
            new_prods = [p * elem for p in start_products]
            start_products.update(new_prods)
        
        # Try aggressive local search with more iterations
        improved_set, improved_products = _local_search(
            start_set, start_products, candidates, deadline, max_swaps=10000
        )
        if len(improved_set) > best_len:
            best_len = len(improved_set)
            best_set = improved_set[:]
    
    # Final validation
    if best_set and not _check_distinct_products(best_set):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return list(best_set)
