import time
import random

def _can_add(products, cand):
    """
    Check if adding cand to a set with given product set keeps all products distinct.
    """
    # Check for collision
    for p in products:
        if p * cand in products:
            return False
    return True

def _add_to_products(products, cand):
    """
    Update products set in place by adding cand.
    """
    new_prods = set()
    for p in products:
        new_prods.add(p * cand)
    products.update(new_prods)

def _local_search(current_set, products, candidates, deadline):
    """
    Try to improve current_set via swap operations.
    """
    k = len(current_set)
    if k >= 18:
        return current_set, products
    
    current_set = list(current_set)
    products = set(products)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set_copy = list(current_set)
        random.shuffle(current_set_copy)
        
        for i in range(len(current_set_copy)):
            if time.time() > deadline:
                break
            removed = current_set_copy[i]
            remaining = [x for x in current_set if x != removed]
            
            # Compute products of remaining
            rem_products = {1}
            for elem in remaining:
                _add_to_products(rem_products, elem)
            
            # Try adding each candidate
            for cand in candidates:
                if cand in current_set:
                    continue
                if _can_add(rem_products, cand):
                    # This swap is valid
                    new_set = remaining + [cand]
                    new_products = set(rem_products)
                    _add_to_products(new_products, cand)
                    return new_set, new_products
        # No swap found in this pass
        # To avoid infinite loop if no improvement is possible, we break
        # But we set improved=False initially, so if we don't return, we break
        if not improved:
            break
            
    return current_set, products

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    rng = random.Random(42)
    
    # Strategy: Random restarts with greedy construction and local search
    for _ in range(50000):
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
            
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        current_set = []
        products = {1}
        
        for cand in shuffled:
            if time.time() > deadline:
                break
            
            if _can_add(products, cand):
                current_set.append(cand)
                _add_to_products(products, cand)
                
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
    
    # Final validation
    if best_set:
        prods = {1}
        valid = True
        for i in best_set:
            new_prods = set()
            for p in prods:
                new_prods.add(p * i)
            if new_prods & prods:
                valid = False
                break
            prods.update(new_prods)
        
        if not valid:
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return best_set
