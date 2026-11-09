"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

DEADLINE = time.time() + 85  # safety margin before 90s kill

def construct():
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute pair_bad[i][j] as a bitmask of k such that {i,j,k} is bad
    # bad if gcd(i,j) == gcd(i,k) == gcd(j,k)
    pair_bad = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcds[i][j]
            mask = 0
            for k in range(n):
                if k == i or k == j: continue
                if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                    mask |= (1 << k)
            pair_bad[i][j] = mask
            pair_bad[j][i] = mask
            
    # Initial state: all numbers in set
    in_set = [True] * n
    current_set_mask = (1 << n) - 1
    
    # Function to check if adding x to current set creates a bad triple
    # We only need to check pairs (i, j) already in the set
    def can_add(x, set_mask):
        # Iterate over all pairs (i, j) in set_mask
        # If x is in pair_bad[i][j], then {i, j, x} is bad
        # To iterate pairs efficiently, we can just iterate over i in set, then j in set (i<j)
        # Since n=128, |S| is at most 128. Pairs ~ 8000.
        # 8000 checks per addition.
        # If we do this for 128 elements, that's ~1M checks. Fast enough.
        
        # Optimization: Only check pairs where at least one is "recently added"? 
        # No, we need global validity.
        
        # Extract indices from set_mask
        # For speed in Python, iterating bits of a 128-bit int is okay.
        # But pair_bad is a list of lists of ints.
        
        # Let's just use a list of indices for the current set for the check
        # This is called frequently, so we should maintain a list.
        pass

    # Strategy: Greedy Removal
    # 1. Count bad triples for each element.
    # 2. Remove element with max count.
    # 3. Update counts.
    
    # Instead of dynamic updates (which are complex), we can do:
    # While there is a bad triple:
    #   Find a bad triple (i, j, k)
    #   Remove the one with the highest "degree" (number of bad triples it's in)
    # This is O(n^3) to find a triple, but n=128 is small.
    
    # However, we can be smarter.
    # Let's use the "Remove Max Conflict" heuristic.
    
    # We will maintain a list `S` of indices currently in the set.
    S = list(range(n))
    S_set = set(S)
    
    # Precompute for each x, the number of bad triples it forms with any pair in S?
    # This is expensive to update.
    
    # Alternative: Iterative improvement.
    # Phase 1: Greedy Construction
    # Start with empty set.
    # Add elements one by one.
    # Order: Try adding elements in an order that minimizes conflicts.
    # A good order: Numbers with many prime factors first? Or few?
    # Numbers with few prime factors (primes) are coprime to many, creating (1,1,1) risks.
    # Numbers with many prime factors (composites) share gcds, creating (d,d,d) risks.
    # Let's try sorting by number of prime factors (Omega(n)).
    
    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
        
    # Order by descending Omega (composites first)
    order = sorted(range(n), key=lambda i: omega(nums[i]), reverse=True)
    
    S = []
    S_set = set()
    
    def can_add_fast(x, S_list):
        # Check if x can be added to S_list
        # For each pair (i, j) in S_list, check if x is in pair_bad[i][j]
        len_S = len(S_list)
        for idx_i in range(len_S):
            i = S_list[idx_i]
            for idx_j in range(idx_i + 1, len_S):
                j = S_list[idx_j]
                if pair_bad[i][j] & (1 << x):
                    return False
        return True

    # Greedy Add
    for x in order:
        if can_add_fast(x, S):
            S.append(x)
            S_set.add(x)
            
    # Phase 2: Local Search / Improvement
    # Try to swap or add.
    # Since we used a specific order, we might have missed some.
    # Try adding any remaining element.
    remaining = [x for x in range(n) if x not in S_set]
    for x in remaining:
        if can_add_fast(x, S):
            S.append(x)
            S_set.add(x)
            
    # Phase 3: Random Restart Local Search (if time permits)
    # The current program did this. I will add a few restarts with different orders.
    best_S = S[:]
    best_len = len(S)
    
    import random
    rng = random.Random(42)
    
    # Try a few different greedy orders
    for restart in range(50):
        if time.time() > DEADLINE:
            break
            
        # Randomize order slightly
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        # Or mix with Omega order
        # Let's just shuffle for now to explore different local optima
        
        S_r = []
        S_r_set = set()
        
        for x in order_rand:
            if can_add_fast(x, S_r):
                S_r.append(x)
                S_r_set.add(x)
                
        # Try to add remaining
        rem = [x for x in range(n) if x not in S_r_set]
        for x in rem:
            if can_add_fast(x, S_r):
                S_r.append(x)
                S_r_set.add(x)
                
        if len(S_r) > best_len:
            best_len = len(S_r)
            best_S = S_r[:]
            
    # Final check and return
    if not best_S:
        return [1]
        
    result = sorted(nums[i] for i in best_S)
    return result
# EVOLVE-BLOCK-END
