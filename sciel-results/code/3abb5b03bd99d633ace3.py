import time
import random
from math import gcd

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 45  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    
    # Heuristic: numbers with larger smallest prime factor or higher values
    # tend to be less "confusable" with products of other numbers.
    # Sort by a score that favors numbers that are "harder" to express as products.
    # We'll use random restarts with different orderings.
    
    def try_greedy(order):
        """Try to build a valid set by greedily adding candidates in given order."""
        current_set = []
        products = {1}
        for cand in order:
            # Check if adding cand keeps all products distinct
            # New products are {p * cand for p in products}
            # Collision if any p * cand is already in products
            valid = True
            for p in products:
                if p * cand in products:
                    valid = False
                    break
            
            if valid:
                current_set.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
        return current_set
    
    def local_search(current_set, remaining_candidates, time_limit):
        """Try to improve the set by swapping or adding elements."""
        if not current_set:
            return current_set
        
        best = current_set[:]
        best_len = len(best)
        products = {1}
        for x in best:
            new_prods = [p * x for p in products]
            products.update(new_prods)
        
        remaining = [c for c in remaining_candidates if c not in best]
        
        # Try to add any remaining candidate
        for cand in remaining:
            if time.time() > time_limit:
                break
            valid = True
            for p in products:
                if p * cand in products:
                    valid = False
                    break
            if valid:
                new_set = best + [cand]
                best = new_set
                best_len += 1
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                remaining = [c for c in remaining if c != cand]
        
        # Try swaps: remove one element, try to add two better ones
        # This is more expensive, so only do it if we have time
        if time.time() < time_limit - 5:
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                # Remove best[i]
                removed = best[i]
                temp_set = best[:i] + best[i+1:]
                # Recompute products for temp_set
                temp_products = {1}
                for x in temp_set:
                    new_prods = [p * x for p in temp_products]
                    temp_products.update(new_prods)
                
                # Try to add candidates from remaining + removed
                add_pool = [c for c in remaining if c != removed]
                add_pool.append(removed)
                random.shuffle(add_pool)
                
                improved = False
                for cand in add_pool:
                    if time.time() > time_limit:
                        break
                    valid = True
                    for p in temp_products:
                        if p * cand in temp_products:
                            valid = False
                            break
                    if valid:
                        # Check if this gives a strictly better set (same size but different, or larger)
                        # For now, just accept if valid and different
                        if cand != removed:
                            new_set = temp_set + [cand]
                            best = new_set
                            products = temp_products
                            new_prods = [p * cand for p in products]
                            products.update(new_prods)
                            remaining = [c for c in remaining if c != cand]
                            improved = True
                            break
                
                if improved:
                    continue
                else:
                    # Restore
                    best = temp_set + [removed]
                    products = temp_products
                    new_prods = [p * removed for p in products]
                    products.update(new_prods)
        
        return best
    
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different ordering strategies per restart
        if r % 3 == 0:
            # Random
            order = candidates[:]
            random.shuffle(order)
        elif r % 3 == 1:
            # Descending (larger numbers first)
            order = sorted(candidates, reverse=True)
            random.shuffle(order[::2])  # Perturb even positions
        else:
            # Ascending with perturbation
            order = sorted(candidates)
            random.shuffle(order[::2])
        
        current_set = try_greedy(order)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            # Try local search to improve
            if time.time() < deadline - 10:
                improved = local_search(best_set, candidates, deadline - 10)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_set = improved
    
    # If no set was found (unlikely), return a valid small set
    if not best_set:
        # Fallback: just primes
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Verify the best_set is valid before returning
    k = len(best_set)
    all_products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= best_set[i]
        if prod in all_products:
            # Collision found, fallback to primes
            return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        all_products.add(prod)
        
    return best_set
