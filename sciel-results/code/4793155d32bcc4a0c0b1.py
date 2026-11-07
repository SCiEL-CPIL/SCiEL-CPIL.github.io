import numpy as np
import time
import sys
from collections import defaultdict
import bisect

def construct():
    # Deadline guard: 85 seconds max (evaluator kills at 90s, leave margin)
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using NumPy sieve
    # sigma[n] = sum of divisors of n
    # Initialize with 1 (divisor 1)
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    # This is O(N log N) but vectorized operations are fast
    for i in range(1, N + 1):
        if time.time() > deadline - 10:
            # Fallback if sieve takes too long
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Group numbers by sigma value
    # Convert to list for faster access in grouping
    sigma_list = sigma.tolist()
    
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma_list[n]
        sigma_groups[s].append(n)
        if n % 1000000 == 0 and time.time() > deadline - 5:
            break

    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair (a, b) such that sigma(a) == sigma(b) and a/b ~ 1.5
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate over groups with at least 2 elements
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
        
        # Sort nums for binary search
        nums_sorted = sorted(nums)
        
        # For each b in nums_sorted, find a in nums_sorted such that a ≈ 3b/2
        # We want to minimize |a/b - 3/2|
        
        # Optimization: Only consider b values that could potentially improve the current best
        # The minimum possible error for a given b is 1/(2b) (when 2a-3b = 1 or -1)
        # If 1/(2b) >= best_error, then no a can improve for this b
        # So we can skip b if b <= 1/(2 * best_error)
        
        # Also, we should prioritize larger b values since they have smaller potential error
        # Let's iterate b from largest to smallest to find good solutions early
        
        # Create a set for O(1) lookup of sigma values? No, we already grouped by sigma.
        # So all nums in this group have the same sigma.
        
        # Iterate b in descending order to find large b (small error) first
        # But we need to check all a for each b?
        # No, for a fixed b, we want a close to 1.5 b.
        # Since nums_sorted is sorted, we can binary search for 1.5 b.
        
        # Let's iterate b in descending order
        for b in reversed(nums_sorted):
            # Pruning: if even the best possible a (error 1/(2b)) is not better than current best, skip
            if 1.0 / (2.0 * b) >= best_error:
                # Since we are iterating b in descending order, smaller b will have even larger min error
                # So we can break out of the b loop for this group?
                # Yes, because b is decreasing, so 1/(2b) is increasing.
                # If 1/(2b) >= best_error, then for all smaller b', 1/(2b') > 1/(2b) >= best_error.
                break
            
            target_a_float = 1.5 * b
            # Binary search for target_a in nums_sorted
            idx = bisect.bisect_left(nums_sorted, int(target_a_float))
            
            # Check candidates around idx
            # We want a such that |2a - 3b| is small.
            # a ≈ 1.5 b.
            # Let's check a window of indices around idx.
            # The error is |2a - 3b| / (2b).
            # For a fixed b, we want to minimize |2a - 3b|.
            # Since a is from nums_sorted, we check neighbors of the ideal a.
            
            found_improvement = False
            for i in range(max(0, idx - 5), min(len(nums_sorted), idx + 6)):
                a = nums_sorted[i]
                if a == b:
                    continue
                
                # Calculate error exactly
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    # Raw value 0 is invalid
                    continue
                
                err_den = 2 * b
                error = err_num / err_den
                
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
                    found_improvement = True
            
            # If we found a very good solution, we might be able to prune more aggressively
            # But the pruning above handles it.

    # Ensure types are Python ints
    return [int(best_a), int(best_b)]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}")
    else:
        print("Type error")
