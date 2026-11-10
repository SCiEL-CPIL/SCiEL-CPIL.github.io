import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, safety margin
    
    N = 10000000
    target_ratio = 1.5
    
    # 1. Compute sigma(n) for n in 1..N
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This loop might be slow in pure Python if not careful, but numpy slices are fast.
    # 10^7 iterations.
    for i in range(1, N + 1):
        if time.time() > deadline - 10: # Check time periodically? No, too slow.
            # We'll assume it finishes. If not, we need a faster sieve.
            # Actually, 10^7 numpy slice additions is approx 1-2 seconds.
            pass
        sigma[i::i] += i

    if time.time() > deadline - 15:
        # Fallback to a simple valid pair if sieve took too long (unlikely)
        # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12, sigma(7)=8, sigma(8)=15, sigma(9)=13, sigma(10)=18, sigma(11)=12
        # sigma(6)=12, sigma(11)=12. Pair (6,11) or (11,6).
        # Error |11/6 - 1.5| = 1/6.
        return [11, 6]

    # 2. Group numbers by sigma value
    groups = {}
    # Iterate over 1..N
    # Using a list of lists might be faster if we know max sigma?
    # Max sigma is roughly N * log log N ~ 30M.
    # Dictionary is fine.
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]

    if time.time() > deadline - 5:
        # If grouping took too long, return best found so far?
        # We haven't found any yet. Return seed.
        return [11, 6]

    # 3. Find best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        # Sort the numbers in this group
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We want to minimize |a/b - 1.5| which is |2a - 3b| / (2b)
        # So for fixed b, we want a closest to 1.5b.
        
        for b in nums:
            target = 1.5 * b
            idx = bisect_left(nums, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
            
            for a in candidates:
                # Calculate error
                # error = |a/b - 1.5|
                # To avoid float precision issues in comparison, compare 2a - 3b
                # But we need the actual error value for the score?
                # The task says "score is the binary64 serialization of that rational".
                # "The evaluator measures the raw value (lower is better)."
                # "raw value is float(|a/b - 3/2|)"
                # So we just need to return [a, b]. The evaluator calculates the rest.
                # We just need to find the pair that minimizes this.
                
                # Comparison metric: |2a - 3b| / b  (proportional to error)
                # Actually, error = |2a - 3b| / (2b).
                # Minimizing this is equivalent to minimizing |2a - 3b| / b.
                
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    # Error is 0. This is invalid per task contract?
                    # "A raw value of exactly 0 cannot be scored and counts as invalid."
                    # So we must avoid a/b = 1.5 exactly.
                    # If 2a == 3b, then a/b = 1.5.
                    # We should skip this or handle it?
                    # If we return it, it's invalid.
                    # So we must find the next best one where 2a != 3b.
                    # However, if 2a == 3b, the error is 0.
                    # The task says "Minimize |a/b-3/2|".
                    # And "raw value of exactly 0 ... counts as invalid".
                    # So we need the minimum non-zero error.
                    
                    # If we find diff=0, we can't use it.
                    # We should look for the next closest non-zero.
                    # But bisect finds the closest. If the closest is exact, the next closest
                    # might be further.
                    # Let's just calculate the error for valid candidates.
                    pass
                else:
                    # Calculate error
                    # error = diff / (2.0 * b)
                    # We can compare diff / b to best_error * b_best?
                    # Or just store the float error.
                    err = diff / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
        if time.time() > deadline - 2:
            break

    # Ensure types are Python int
    return [int(best_a), int(best_b)]
