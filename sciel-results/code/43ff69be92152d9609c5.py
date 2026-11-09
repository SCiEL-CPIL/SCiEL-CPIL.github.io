import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    
    # 1. Sieve for primes and prime factors
    # We need to know if a number has any prime factor > 128
    # And we need the mask of small primes for the B&B
    
    # Primes up to 256
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Split primes
    small_primes = [p for p in primes if p <= 128]
    large_primes = [p for p in primes if p > 128]
    
    # Map small primes to bit indices
    prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    
    # Calculate prime mask for each number 1..256
    # mask[n] = bitmask of small prime factors of n
    # If n has a large prime factor, we mark it as "free" or exclude from contested set
    # Actually, if n has a large prime factor, n must be that large prime (since 2*129 > 256)
    # So we can just check if n is in large_primes.
    
    prime_masks = [0] * (N + 1)
    has_large_prime = [False] * (N + 1)
    
    for n in range(2, N + 1):
        temp = n
        mask = 0
        is_large = False
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime factor
            if temp > 128:
                is_large = True
            else:
                # temp is a small prime (should be in small_primes)
                mask |= (1 << prime_to_idx[temp])
                
        prime_masks[n] = mask
        has_large_prime[n] = is_large

    # 2. Determine Fixed Selection
    # 1 is always selected
    # All large primes are always selected
    base_sum = 1
    for p in large_primes:
        base_sum += p
        
    # 3. Determine Contested Candidates
    # Numbers n in [2, 256] that do NOT have a large prime factor
    # AND n != 1
    candidates = []
    for n in range(2, N + 1):
        if not has_large_prime[n]:
            # This number is composed only of small primes
            # It is a candidate for the B&B
            candidates.append((n, prime_masks[n]))
            
    # Sort by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Branch and Bound
    # We want to maximize sum of selected candidates
    # Subject to: no two selected candidates share a prime factor
    
    best_total = base_sum
    n_cands = len(candidates)
    
    # Precompute suffix sums of weights for a quick upper bound?
    # sum_weights[i] = sum of values of candidates[i:]
    # But this ignores conflicts. It's a valid upper bound though.
    suffix_sums = [0] * (n_cands + 1)
    for i in range(n_cands - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + candidates[i][0]
        
    # Greedy initial solution to tighten bound
    # Greedy: pick largest compatible
    greedy_mask = 0
    greedy_sum = 0
    for val, mask in candidates:
        if greedy_mask & mask == 0:
            greedy_mask |= mask
            greedy_sum += val
    best_total = max(best_total, base_sum + greedy_sum)
    
    # B&B Function
    # idx: current index in candidates
    # current_sum: sum of selected values so far (excluding base)
    # used_mask: bitmask of used small primes
    
    def search(idx, current_sum, used_mask):
        nonlocal best_total
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: Upper Bound
        # The maximum possible addition is the sum of all remaining candidates
        # (even if they conflict, they can't exceed this sum)
        # A tighter bound: sum of candidates j >= idx that are compatible with used_mask
        # This is O(N) per call.
        
        # Quick check with suffix sum first
        if current_sum + suffix_sums[idx] <= best_total - base_sum:
            return
            
        # Refine bound:
        potential = 0
        for j in range(idx, n_cands):
            val, mask = candidates[j]
            if mask & used_mask == 0:
                potential += val
            # If potential + current_sum <= best, we could break early?
            # But we need the sum of ALL compatible ones for a valid upper bound.
            # Actually, if we just sum all compatible, it's a valid upper bound.
            
        if current_sum + potential <= best_total - base_sum:
            return
            
        # Iterate
        for i in range(idx, n_cands):
            val, mask = candidates[i]
            
            # If incompatible, skip (implicitly excluded)
            if mask & used_mask != 0:
                continue
                
            # Option 1: Include candidates[i]
            new_sum = current_sum + val
            if new_sum > best_total - base_sum:
                best_total = base_sum + new_sum
                
            new_mask = used_mask | mask
            
            # Recurse
            search(i + 1, new_sum, new_mask)
            
            # Option 2: Exclude candidates[i]
            # This is handled by the loop continuing to i+1
            # But we must ensure we don't miss combinations where we exclude i
            # and include j > i. The loop structure `for i in range...`
            # combined with `search(i+1...)` for inclusion
            # and simply continuing the loop for exclusion
            # is correct.
            
            # Optimization: If we included i, we explored all subsets of {i+1..} compatible with i.
            # Now we continue the loop, which explores subsets of {i+1..} NOT containing i
            # (and compatible with previous choices).
            # This covers all cases.
            
            # Check time
            if time.time() > deadline:
                return
                
    search(0, 0, 0)
    
    # Convert to list of ints
    # We lost the actual set in the simple B&B above, only the sum.
    # We need to return the list.
    # We need to store the best set.
    
    # Refined B&B to track the set
    best_set = []
    
    def search2(idx, current_sum, used_mask, current_set):
        nonlocal best_total, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix_sums[idx] <= best_total - base_sum:
            return
            
        potential = 0
        for j in range(idx, n_cands):
            val, mask = candidates[j]
            if mask & used_mask == 0:
                potential += val
                
        if current_sum + potential <= best_total - base_sum:
            return
            
        for i in range(idx, n_cands):
            val, mask = candidates[i]
            if mask & used_mask != 0:
                continue
                
            new_sum = current_sum + val
            new_set = current_set + [val]
            
            if new_sum > best_total - base_sum:
                best_total = base_sum + new_sum
                best_set = new_set[:]
                
            new_mask = used_mask | mask
            search2(i + 1, new_sum, new_mask, new_set)
            
            if time.time() > deadline:
                return
                
    search2(0, 0, 0, [])
    
    # Construct final list
    # base_sum consists of 1 and large primes.
    # We need to reconstruct the large primes list.
    result = [1] + large_primes + best_set
    
    # Ensure types are int
    return [int(x) for x in result]

# Test locally? No, just output.
