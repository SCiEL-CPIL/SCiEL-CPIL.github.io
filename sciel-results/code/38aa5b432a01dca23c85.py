"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precompute sigma(n) for all n up to 10^7 using a sieve
    # This is O(N log N) and fast enough in Python with list operations or simple loops.
    # A standard sieve for sigma:
    # sigma(n) = sum of divisors.
    # We can do this by iterating i from 1 to N, and adding i to sigma[j] for all multiples j of i.
    # This is sum_{i=1}^N N/i = N * H_N ~ N log N.
    # For N=10^7, this is about 10^7 * 16 ~ 1.6 * 10^8 operations.
    # In Python, a nested loop like this might be too slow (1.6s per 10^7 ops? No, Python is slower).
    # 1.6 * 10^8 simple integer additions in Python might take ~10-20 seconds.
    # Let's try to optimize or check if it fits.
    
    MAX_N = 10000000
    sigma_arr = [0] * (MAX_N + 1)
    
    # Sieve for sigma
    # i is the divisor
    for i in range(1, MAX_N + 1):
        # j is the multiple
        # j = i, 2i, 3i, ...
        for j in range(i, MAX_N + 1, i):
            sigma_arr[j] += i
            
    if time.time() > deadline:
        # Fallback if sieve takes too long (unlikely if optimized, but safe)
        # Return a known valid pair
        return [11, 6]

    # 2. Group numbers by their sigma value
    # fibers[sigma_val] = list of n such that sigma(n) == sigma_val
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma_arr[n]].append(n)
        
    if time.time() > deadline:
        return [11, 6]

    # 3. Search for pairs (a, b) in the same fiber minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want small |2a - 3b| and large b.
    # Let's iterate through the fibers.
    # For each fiber, we have a list of numbers L.
    # We want to find a, b in L such that 2a ≈ 3b.
    # This is equivalent to a ≈ 1.5 b.
    # We can sort L.
    # For each b in L, we can binary search for a in L closest to 1.5 b.
    # Since we want to minimize error, and error = |2a - 3b| / (2b),
    # large b is good. So we should process larger b first?
    # Or just check all candidates and keep the best.
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)
    
    # To make the search efficient, we can iterate through fibers.
    # How many fibers? sigma(n) can be up to ~ n log log n.
    # Many numbers will have unique sigma values.
    # We only care about fibers with size > 1.
    
    # Optimization:
    # Instead of checking every b in every fiber, we can be smarter.
    # But let's first just do the binary search approach for all fibers with size > 1.
    # Total number of elements in all fibers is MAX_N.
    # If we sort each fiber, total sorting cost is sum(|L_i|^2) in worst case? No, sum(|L_i| log |L_i|).
    # Most fibers are small.
    
    # Let's iterate through the fibers dict.
    for s_val, nums in fibers.items():
        if time.time() > deadline:
            break
            
        if len(nums) < 2:
            continue
            
        # Sort the list of numbers in this fiber
        # We want to find a, b in nums such that a/b is close to 1.5
        # Let's sort nums.
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We can use two pointers or binary search.
        # Since we want to minimize |a/b - 1.5|, and a, b are from the same set,
        # and the set is sorted, we can use a two-pointer technique?
        # Or simply for each b, bisect 1.5*b.
        
        # To prioritize large b (which reduces the denominator of the error term),
        # we could iterate b in descending order?
        # But we need to check all to be sure.
        # However, if we find a very small error, we can stop early?
        # The theoretical min error is 0 (if 2a=3b).
        # If 2a=3b, then a=1.5b. Since a,b are integers, b must be even, a=3k, b=2k.
        # If such a pair exists in a fiber, error is 0.
        
        # Let's just iterate through all b in the sorted list.
        # To speed up, we can stop if the current best_err is extremely small.
        
        # Binary search for each b
        # nums is sorted ascending.
        # target_a = 1.5 * b
        # We want a in nums closest to target_a.
        
        # We can use bisect
        import bisect
        
        for b in nums:
            if time.time() > deadline:
                break
                
            target_a = 1.5 * b
            # Find insertion point
            pos = bisect.bisect_left(nums, target_a)
            
            # Check neighbors: pos-1, pos, pos+1
            # pos is where target_a would be inserted to keep list sorted.
            # So nums[pos-1] < target_a <= nums[pos] (if pos < len)
            # Candidates: pos-1, pos
            
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            # Also check pos+1 just in case? No, closest is either floor or ceil.
            
            for a in candidates:
                if a == b:
                    continue # a and b must be distinct? 
                    # The problem says "Approximate a ratio... sigma(a)=sigma(b)".
                    # It doesn't explicitly say a != b.
                    # If a == b, a/b = 1, |1 - 1.5| = 0.5.
                    # But usually these tasks imply distinct or just any pair.
                    # If a=b is allowed, and we find a pair with a=b, error is 0.5.
                    # But we can likely do better.
                    # Let's assume a and b can be equal, but it's unlikely to be optimal.
                    
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
                    # If we hit 0, we can't do better.
                    if err == 0.0:
                        return [int(best_a), int(best_b)]
                        
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
