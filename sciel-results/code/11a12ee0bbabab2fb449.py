import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [x for x in range(2, 41) if x not in primes]
    
    # Initial set is all primes
    S = list(primes)
    
    # Calculate initial products
    P = {1}
    for p in primes:
        new_vals = [p * x for x in P]
        P.update(new_vals)
        
    best_size = len(S)
    best_set = list(S)
    
    # DFS to add composites
    # We sort composites. Maybe larger ones first? Or smaller?
    # Smaller ones are more likely to be ratios of smaller products.
    # Larger ones are less likely to be in P(S) or ratios.
    # Let's try sorting descending to find a large set quickly?
    # Or ascending to prune?
    # Let's just try all permutations? No, 27! is too big.
    # Greedy with backtracking.
    
    def can_add(c, P):
        # Check if c is a ratio of any two elements in P
        # c = p2 / p1  => p2 = c * p1
        # So check if c * p1 is in P for any p1 in P
        for p1 in P:
            if c * p1 in P:
                return False
        return True

    def dfs(current_S, current_P, candidates):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_set = list(current_S)
            
        if not candidates:
            return
            
        # Try adding each candidate
        for i, c in enumerate(candidates):
            if not can_add(c, current_P):
                continue
                
            # Add c
            new_P = set(current_P)
            for p in current_P:
                new_P.add(c * p)
                
            new_S = current_S + [c]
            new_candidates = candidates[:i] + candidates[i+1:]
            
            dfs(new_S, new_P, new_candidates)
            
            if time.time() > deadline:
                return

    # Heuristic: Sort candidates. 
    # Composites that are powers of primes (4, 8, 9, 16, 25, 27, 32) might be safer?
    # Or those with large prime factors?
    # Let's just use the list as is, or sorted.
    # To maximize chance of finding a large set, we should explore promising branches first.
    # Composites that are "far" from being ratios are better.
    
    # Let's try a specific ordering: 
    # 1. Composites that are not divisible by any prime in S? Impossible, all are.
    # 2. Composites with large values?
    
    # Let's try sorting by value descending.
    sorted_composites = sorted(composites, reverse=True)
    
    dfs(S, P, sorted_composites)
    
    # Verify result
    result = [int(x) for x in best_set]
    if not _verify(result):
         # Fallback to primes if verification fails (should not happen)
         return primes
         
    return result

def _verify(candidate):
    """Verify that all subset products are distinct."""
    k = len(candidate)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= candidate[i]
        if prod in products:
            return False
        products.add(prod)
    return True
