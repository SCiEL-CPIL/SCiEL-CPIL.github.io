"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_sieve_and_masks(max_val):
    """Precompute smallest prime factor and bitmask of prime factors for sums up to max_val."""
    # We need masks for sums up to 2 * 10000 = 20000
    limit = max_val
    spf = list(range(limit + 1))
    primes = []
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            primes.append(i)
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    # Add remaining primes
    for i in range(int(math.isqrt(limit)) + 1, limit + 1):
        if spf[i] == i:
            primes.append(i)
            
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * (limit + 1)
    for n in range(2, limit + 1):
        m = n
        mask = 0
        while m > 1:
            p = spf[m]
            mask |= (1 << prime_to_idx[p])
            while m % p == 0:
                m //= p
        masks[n] = mask
        
    return masks

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    max_val = 10000
    k = 12
    sum_limit = 2 * max_val
    
    # Precompute masks for all possible sums
    masks = _build_sieve_and_masks(sum_limit)
    
    # Initial guess: Try to find a set with few primes
    # Start with small even numbers
    best_numbers = list(range(2, 2 + 2 * k, 2)) # 2, 4, ..., 24
    # Ensure distinct and <= max_val
    best_numbers = sorted(set(best_numbers))
    if len(best_numbers) < k:
        # Fallback
        best_numbers = list(range(1, k + 1))
        
    # Calculate initial count
    def get_count(nums):
        total_mask = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                total_mask |= masks[s]
        return total_mask.bit_count(), total_mask

    current_count, current_mask = get_count(best_numbers)
    best_count = current_count
    best_mask = current_mask

    # Local Search
    # We will try to improve by swapping one element
    # To speed up, we precompute the mask of all pairs NOT involving index i
    # mask_fixed[i] = OR of masks[x_u + x_v] for u, v != i
    
    n = k
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Precompute fixed masks for each index
        fixed_masks = [0] * n
        # For each i, fixed_mask is OR of all pairs (u,v) where u,v != i
        # Total pairs = n*(n-1)/2
        # For a specific i, there are (n-1)*(n-2)/2 pairs not involving i
        # We can compute this efficiently:
        # total_mask_all = OR of all pairs
        # But we need the OR of a subset.
        # Since n=12, we can just loop.
        
        # Optimization: 
        # For each i, we want to replace x_i with x_new.
        # The new mask will be fixed_mask[i] | new_pairs_mask(i, x_new)
        # fixed_mask[i] is the OR of masks[x_u + x_v] for u,v != i.
        
        for i in range(n):
            # Compute fixed_mask for index i
            fm = 0
            for u in range(n):
                if u == i: continue
                for v in range(u + 1, n):
                    if v == i: continue
                    fm |= masks[best_numbers[u] + best_numbers[v]]
            
            # Try replacing best_numbers[i]
            current_val = best_numbers[i]
            
            # Candidate values: 
            # We can try all values 1..max_val, but that's 10000.
            # Let's try a random subset or all if fast enough.
            # 10000 * 11 lookups = 110k ops per i. 12 * 110k = 1.3M ops per pass.
            # In Python, this might take ~0.1s - 0.5s per pass.
            # We have ~80s. So ~200-800 passes. Should be enough.
            
            best_new_val = None
            best_new_mask = None
            best_new_count = current_count # We want strictly less
            
            # Iterate through possible new values
            # To make it faster, we can skip values that are already in best_numbers
            existing = set(best_numbers)
            
            for new_val in range(1, max_val + 1):
                if new_val in existing:
                    continue
                
                # Calculate mask for pairs involving i with new_val
                nm = 0
                for j in range(n):
                    if j == i: continue
                    nm |= masks[new_val + best_numbers[j]]
                
                total_new_mask = fm | nm
                cnt = total_new_mask.bit_count()
                
                if cnt < best_new_count:
                    best_new_count = cnt
                    best_new_val = new_val
                    best_new_mask = total_new_mask
                    if cnt == 1: # Theoretical minimum
                        break
                
            if best_new_val is not None:
                # Apply change
                best_numbers[i] = best_new_val
                best_numbers = sorted(best_numbers) # Keep sorted for consistency? 
                # Actually, indices change if we sort. 
                # It's better to keep the list unsorted during search to avoid index shifting,
                # or re-sort and restart the loop.
                # Let's just update the value and keep the position.
                # The problem doesn't require sorted output, just distinct integers.
                # But _count_distinct_primes in seed used sorted.
                # The evaluator just checks the set.
                # So we can keep the list unsorted.
                
                current_count = best_new_count
                current_mask = best_new_mask
                improved = True
                # Break out of i-loop to recalculate fixed masks?
                # Yes, fixed masks depend on the values.
                break # Break i loop, restart while loop
        
    # Final verification and return
    # Ensure distinctness and bounds
    final_nums = sorted(set(best_numbers))
    if len(final_nums) != k:
        # Should not happen
        return list(range(1, k+1))
        
    # The task asks to return n=12 distinct positive integers.
    return final_nums

# EVOLVE-BLOCK-END
