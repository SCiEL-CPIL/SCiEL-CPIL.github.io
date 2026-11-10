import time
import random

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses packed prime-exponent vectors for fast collision checking.
    """
    deadline = time.time() + 45  # Safety margin before 60s limit
    best_set = []
    best_len = 0
    
    candidates = list(range(2, 41))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Precompute packed exponent vectors (7 bits per prime field, enough for sums up to 127)
    def factorize(n):
        vec = 0
        offset = 0
        for i, p in enumerate(primes):
            if p > n:
                break
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            vec |= (exp << offset)
            offset += 7
        return vec
    
    packed = {}
    for c in candidates:
        packed[c] = factorize(c)
    
    def greedy_build(order):
        """Greedy build using a given ordering. Returns (set, products_set)."""
        current_set = []
        current_products = {0}  # empty product has exponent vector 0
        
        for cand in order:
            cand_packed = packed[cand]
            
            # Check collision: is (p + cand_packed) in current_products for any p?
            collision = False
            for p in current_products:
                if (p + cand_packed) in current_products:
                    collision = True
                    break
            
            if not collision:
                current_set.append(cand)
                new_prods = [p + cand_packed for p in current_products]
                current_products.update(new_prods)
        
        return current_set, current_products
    
    # Try deterministic orderings first
    # 1. Sorted ascending
    set1, _ = greedy_build(sorted(candidates))
    if len(set1) > best_len:
        best_len = len(set1)
        best_set = set1[:]
    
    # 2. Sorted descending
    set2, _ = greedy_build(sorted(candidates, reverse=True))
    if len(set2) > best_len:
        best_len = len(set2)
        best_set = set2[:]
    
    # 3. Primes first, then composites
    primes_list = [c for c in candidates if all(c % p != 0 for p in primes if p < c)]
    composites = [c for c in candidates if c not in primes_list]
    set3, _ = greedy_build(primes_list + composites)
    if len(set3) > best_len:
        best_len = len(set3)
        best_set = set3[:]
    
    # Random restarts
    num_restarts = 100000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set, _ = greedy_build(shuffled)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
            # If we found 18 or more, that's excellent
            if best_len >= 18:
                # Keep searching briefly for potentially larger sets
                pass
    
    # If no set was found (unlikely), return a valid small set
    if not best_set:
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return best_set
