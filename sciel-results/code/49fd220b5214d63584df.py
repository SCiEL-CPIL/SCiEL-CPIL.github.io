import time
from math import isqrt

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin for 90s limit
    
    N = 256
    
    # 1. Find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute factor masks for each number 1..N
    # mask[x] is a bitmask of prime indices dividing x
    masks = [0] * (N + 1)
    for x in range(2, N + 1):
        m = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                m |= (1 << prime_to_idx[temp])
            else:
                # Should not happen if primes list is complete
                pass
        masks[x] = m
    masks[1] = 0
    
    # 3. Precompute candidates for each prime
    # candidates[p_idx] = list of numbers x such that p_idx is in masks[x]
    candidates = [[] for _ in range(num_primes)]
    for x in range(2, N + 1):
        m = masks[x]
        # Iterate through set bits in m
        mm = m
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            candidates[idx].append(x)
            mm ^= lsb
            
    # Sort candidates for each prime by value descending
    for idx in range(num_primes):
        candidates[idx].sort(reverse=True)
        
    # 4. Initial Best Solution
    # Start with Seed: Primes + 1
    seed_sum = 1 + sum(primes)
    best_sum = seed_sum
    
    # Try a simple greedy to get a better initial bound
    # Greedy: Pick largest x, remove its factors, repeat
    def greedy():
        avail_mask = (1 << num_primes) - 1
        total = 1
        # Create a list of numbers to consider, sorted by value desc
        nums = list(range(N, 1, -1))
        for x in nums:
            if time.time() > deadline:
                break
            m = masks[x]
            if m & ~avail_mask == 0: # All factors available
                total += x
                avail_mask &= ~m
        return total
        
    g_sum = greedy()
    if g_sum > best_sum:
        best_sum = g_sum
        
    # 5. Branch and Bound
    # State: mask (available primes), current_sum
    # We use a non-recursive approach or careful recursion to avoid stack depth issues?
    # Depth is at most num_primes (54), so recursion is fine.
    
    def upper_bound(mask, current_sum):
        ub = current_sum
        # For each prime in mask, find max x
        # To speed up, we can just iterate primes in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # Find max x in candidates[idx] such that masks[x] is subset of mask
            max_val = 0
            for x in candidates[idx]:
                if masks[x] & ~mask == 0:
                    if x > max_val:
                        max_val = x
                    else:
                        break # Since sorted desc, if we found one, next are smaller? 
                              # Wait, we break if x <= max_val? No, we want the FIRST one that fits.
                              # Since sorted desc, the first one that fits is the max.
                              # So we can just return the first fit.
            ub += max_val
            m ^= lsb
        return ub
        
    def search(mask, current_sum):
        nonlocal best_sum
        
        if time.time() > deadline:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            
        if mask == 0:
            return
            
        # Calculate Upper Bound
        ub = upper_bound(mask, current_sum)
        if ub <= best_sum:
            return
            
        # Pick the lowest set bit (smallest prime index)
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        bit = 1 << idx
        
        # Branch 1: Do not use this prime
        search(mask & ~bit, current_sum)
        
        # Branch 2: Use this prime in some number x
        # Iterate candidates for this prime
        # We should iterate in a way that finds good solutions early.
        # Candidates are already sorted by value desc.
        for x in candidates[idx]:
            if time.time() > deadline:
                return
            m_x = masks[x]
            # Check if all factors of x are available in mask
            if m_x & ~mask == 0:
                new_mask = mask & ~m_x
                search(new_mask, current_sum + x)
                
    # Start search with all primes available
    # Note: We don't need to explicitly handle 1 in the mask logic, 
    # as 1 has no prime factors and adds 1 to the sum.
    # We included 1 in initial sums (seed_sum, greedy).
    # In the search, we start with current_sum = 0? 
    # No, we should start with current_sum = 1 (for the number 1) 
    # and mask = all primes.
    
    initial_mask = (1 << num_primes) - 1
    search(initial_mask, 1)
    
    # Construct the result? 
    # The task asks to return a LIST of ints.
    # The B&B above only tracks the SUM. 
    # We need to recover the SET of numbers.
    
    # This is a problem. We need to store the path.
    # Let's modify the search to store the best set.
    
    # Re-do with path tracking.
    
    best_set = [1] + primes # Fallback to seed
    
    def upper_bound_mask(mask):
        # Returns just the additional sum potential
        ub = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            max_val = 0
            for x in candidates[idx]:
                if masks[x] & ~mask == 0:
                    max_val = x
                    break
            ub += max_val
            m ^= lsb
        return ub

    def search_with_path(mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        if mask == 0:
            return
            
        ub = current_sum + upper_bound_mask(mask)
        if ub <= best_sum:
            return
            
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        bit = 1 << idx
        
        # Branch 1: Skip prime
        search_with_path(mask & ~bit, current_sum, current_set)
        
        # Branch 2: Use prime in x
        for x in candidates[idx]:
            if time.time() > deadline:
                return
            m_x = masks[x]
            if m_x & ~mask == 0:
                new_mask = mask & ~m_x
                # Append x to set
                # To avoid copying list every time, we can pass a tuple or just copy.
                # Depth is small, copying is fine.
                search_with_path(new_mask, current_sum + x, current_set + [x])
                
    search_with_path(initial_mask, 1, [1])
    
    # Ensure types are correct
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
