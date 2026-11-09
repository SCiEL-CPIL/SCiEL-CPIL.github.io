import math
import random
import time

def construct():
    max_mod = 128
    candidates = list(range(2, max_mod + 1))
    
    # Precompute gcd for all pairs of candidates to speed up checks?
    # 127 * 127 / 2 ~ 8000 pairs. Trivial.
    # But we need gcd(m1, m2) for dynamic subsets.
    # math.gcd is fast enough.
    
    def is_compatible(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def fast_greedy(order):
        current_pairs = [] # list of (m, r)
        
        for m in order:
            found = False
            # Try residues. 
            # Optimization: If m is large, maybe try random residues first? 
            # No, deterministic is safer for finding *a* solution.
            # But we want to find *the* solution that allows future additions.
            # Greedy is myopic.
            
            for r in range(m):
                valid = True
                for m2, r2 in current_pairs:
                    if not is_compatible(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    current_pairs.append((m, r))
                    found = True
                    break
            
        return current_pairs

    best_size = 0
    best_pairs = []
    
    # Generate different orders
    orders = []
    
    # 1. Random shuffles
    # 2. Sorted by number of prime factors (descending) - "Hub" nodes first
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    # Precompute factor counts
    factor_counts = {m: num_prime_factors(m) for m in candidates}
    
    # Order 1: Descending factor count
    orders.append(sorted(candidates, key=lambda x: factor_counts[x], reverse=True))
    # Order 2: Ascending factor count
    orders.append(sorted(candidates, key=lambda x: factor_counts[x]))
    # Order 3: Ascending value
    orders.append(sorted(candidates))
    # Order 4: Descending value
    orders.append(sorted(candidates, reverse=True))
    
    # Add some random shuffles
    for _ in range(50):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    start_time = time.time()
    
    for order in orders:
        if time.time() - start_time > 55:
            break
            
        pairs = fast_greedy(order)
        if len(pairs) > best_size:
            best_size = len(pairs)
            best_pairs = pairs
            
    # Convert to required format
    result = [[r, m] for m, r in best_pairs]
    return result
