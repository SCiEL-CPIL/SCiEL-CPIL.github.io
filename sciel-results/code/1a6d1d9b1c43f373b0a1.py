import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 10_000_000
    
    # 1. Sieve for sigma using NumPy for speed
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # The loop for d in 1..N
    # This is the bottleneck.
    # We can check time every 10000 iterations of d?
    # Or just let it run. If it exceeds, we return current best?
    # But we need the full sigma array to group.
    # If we can't finish sieve, we can't do the grouping.
    # Fallback: Return a known good pair if sieve is too slow?
    # Seed score ~4e6 implies error ~2.5e-7
    # We can try to find a good pair in smaller range?
    # Or just return 11, 6?
    # 11/6 error is 1/3. Score 3.
    # That's much worse than 4e6.
    # If we time out here, we return invalid or low score.
    # Let's try to be robust.
    
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 100000 == 0:
            if time.time() - t0 > DEADLINE:
                # Fallback
                # Use a known good pair from seed or small search
                # Seed score ~4e6 implies error ~2.5e-7
                # We can try to find a good pair in smaller range?
                # Or just return 11, 6?
                # 11/6 error is 1/3. Score 3.
                # That's much worse than 4e6.
                # If we time out here, we return invalid or low score.
                # Let's try to be robust.
                # We need to return something valid.
                # 11, 6: sigma(11)=12, sigma(6)=12. 11/6 = 1.833...
                # |1.833 - 1.5| = 0.333
                # Score = 1/0.333 = 3
                # This is a valid fallback.
                return [11, 6]
    
    # 2. Group by sigma value
    # For efficiency, only consider numbers where a/b is close to 3/2
    # a ≈ 3b/2, so for each b, a should be near 3b/2
    
    # Create a dict: sigma_value -> list of n with that sigma
    # But this could use a lot of memory. Let's be smarter.
    
    # For each b from 1 to N, target a ≈ 3b/2
    # Check if there exists a near 3b/2 with sigma(a) = sigma(b)
    
    # Build lookup: for each sigma value, store the numbers
    # Use a dictionary
    sigma_to_nums = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_nums[int(sigma[n])].append(n)
    
    # Now search for best pair
    best_err = float('inf')
    best_pair = (11, 6)  # fallback
    
    # For b in range(1, N+1):
    #   target_a = 3*b/2
    #   Check a values near target_a in sigma_to_nums[sigma[b]]
    
    # To be efficient, for each sigma group, sort the numbers
    # Then for each b, binary search for a near 3b/2
    
    # But iterating all b is O(N) and for each we do a binary search - should be OK
    
    # Actually, let's iterate over b and check a few candidates near 3b/2
    # For each b, a must satisfy sigma(a) = sigma(b) and a is near 3b/2
    
    # Let's collect all numbers by sigma value, sorted
    for s_val, nums in sigma_to_nums.items():
        if len(nums) < 2:
            continue
        if time.time() - t0 > DEADLINE:
            break
        nums.sort()
        # For each b in nums, find a in nums closest to 3b/2
        for b in nums:
            target_a = 3 * b / 2.0
            # Binary search for target_a in nums
            idx = bisect.bisect_left(nums, target_a)
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
            for a in candidates:
                if a == b:
                    continue
                if a > N or b > N:
                    continue
                # Compute exact rational error
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    continue
                err = num / den
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        return [int(best_pair[0]), int(best_pair[1])]
    
    return [int(best_pair[0]), int(best_pair[1])]
