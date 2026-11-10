import time
import sys
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000
    
    # Use bytearray for is_comp to save memory and improve cache performance
    is_comp = bytearray(MAX_N + 1)
    sigma = [0] * (MAX_N + 1)
    spow = [0] * (MAX_N + 1)
    sval = [0] * (MAX_N + 1)
    primes = []
    
    sigma[1] = 1
    spow[1] = 1
    sval[1] = 1
    
    # Linear Sieve for sigma
    for i in range(2, MAX_N + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            sval[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            j = i * p
            if j > MAX_N:
                break
            is_comp[j] = 1
            if i % p == 0:
                spow[j] = spow[i] * p
                sval[j] = sval[i] + spow[j]
                sigma[j] = (sigma[i] // sval[i]) * sval[j]
                break
            else:
                spow[j] = p
                sval[j] = 1 + p
                sigma[j] = sigma[i] * (1 + p)
                
    if time.time() > deadline - 15:
        # Fallback if sieve takes too long (unlikely for 10^7)
        return [11, 6]

    # Group by sigma
    # Instead of defaultdict, we can sort pairs to avoid overhead?
    # defaultdict is usually fast enough.
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
        
    if time.time() > deadline - 10:
        return [11, 6]

    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5) # 0.25
    
    # Process fibers
    # Heuristic: Process fibers with larger numbers first?
    # We can sort the keys by max value in the fiber, descending.
    # This might help find a good (large b) solution early.
    
    # To sort keys by max value, we need to compute max for each key.
    # This is O(sum of fiber sizes) = O(N).
    # Then sort keys: O(K log K) where K is number of unique sigmas.
    
    # However, sorting the keys might take time. 
    # Alternative: Just iterate. The binary search is fast.
    # Let's try iterating as is, but add a check for early exit if best_err is very small?
    # Or, we can sort the fibers by the maximum element in the fiber, descending.
    
    # Let's create a list of (max_val, fiber_list)
    fiber_list = []
    for s_val, nums in fibers.items():
        if len(nums) < 2:
            continue
        max_val = max(nums)
        fiber_list.append((max_val, s_val, nums))
        
    # Sort by max_val descending
    fiber_list.sort(key=lambda x: x[0], reverse=True)
    
    if time.time() > deadline - 5:
        return [int(best_a), int(best_b)]

    for max_val, s_val, nums in fiber_list:
        if time.time() > deadline - 1:
            break
            
        # Sort nums for binary search
        nums.sort()
        
        # For each b in nums, find a closest to 1.5 * b
        for b in nums:
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            
            # Check pos and pos-1
            if pos < len(nums):
                a = nums[pos]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        if err < 1e-9: # Very small, likely 0
                            return [int(best_a), int(best_b)]
            
            if pos > 0:
                a = nums[pos-1]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        if err < 1e-9:
                            return [int(best_a), int(best_b)]
                            
    return [int(best_a), int(best_b)]
