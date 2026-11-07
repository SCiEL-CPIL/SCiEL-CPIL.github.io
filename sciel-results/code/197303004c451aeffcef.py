"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    MAX_N = 10_000_000
    
    # Step 1: Compute sigma(n) for all n up to MAX_N using a sieve
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    # Check time
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Step 2: Group numbers by sigma value
    # Use a dict: sigma_value -> list of numbers
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)
    
    # Check time
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Step 3: For each group, find pair (a, b) minimizing |2a - 3b|
    # We want a/b ≈ 3/2, so 2a ≈ 3b
    # For a given b, optimal a ≈ 3b/2
    # Use two-pointer: sort group, for each b, find a closest to 3b/2
    
    best_val = float('inf')
    best_a, best_b = 11, 6  # fallback: sigma(11)=12, sigma(6)=12
    
    for s_val, nums in groups.items():
        if len(nums) < 2:
            continue
        
        nums.sort()
        n = len(nums)
        
        # Two-pointer approach:
        # For each b in nums, we want a such that 2a ≈ 3b, i.e., a ≈ 1.5*b
        # We can use a pointer that moves as b increases
        # Since we want |2a - 3b| minimized, for fixed b, optimal a is closest to 1.5*b
        
        # Use binary search for each b to find a closest to 1.5*b
        import bisect
        
        for i in range(n):
            b = nums[i]
            target_a = 1.5 * b
            # Find position where target_a would be in nums
            pos = bisect.bisect_left(nums, int(target_a))
            # Check pos and pos-1 (and maybe pos+1 for safety)
            candidates = []
            for p in [pos - 1, pos, pos + 1]:
                if 0 <= p < n:
                    candidates.append(nums[p])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                # We minimize |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Can't have exactly 0
                val = diff / (2 * b)
                if val < best_val:
                    best_val = val
                    best_a, best_b = a, b
        
        if time.time() > deadline - 5:
            break
    
    # Ensure we don't return a pair where a == b (which gives 0)
    if best_a == best_b:
        best_a, best_b = 11, 6
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
