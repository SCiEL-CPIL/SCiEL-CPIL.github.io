"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

MAX_N = 10000000
TARGET_NUM = 3
TARGET_DEN = 2

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Sieve for sigma(n)
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
    
    # Find best pair with a/b close to 3/2
    best_error = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12
    
    for s, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        # For each b in nums, check if 3*b/2 is close to some a in nums
        # We want a/b ≈ 3/2, so a ≈ 3*b/2
        for b in nums:
            target_a = 3 * b / 2
            # Binary search in sorted nums for closest to target_a
            # nums is already sorted since we appended in order
            lo, hi = 0, len(nums) - 1
            best_a_for_b = None
            best_err_for_b = None
            # Check neighbors of target_a
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] == int(target_a):
                    best_a_for_b = nums[mid]
                    best_err_for_b = abs(nums[mid] / b - 1.5)
                    lo = mid + 1
                elif nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            # Check the two closest elements
            for idx in [lo - 1, lo]:
                if 0 <= idx < len(nums):
                    a = nums[idx]
                    if a == b:
                        continue
                    err = abs(a / b - 1.5)
                    if err == 0:
                        continue
                    if best_err_for_b is None or err < best_err_for_b:
                        best_err_for_b = err
                        best_a_for_b = a
            
            if best_a_for_b is not None and best_err_for_b is not None:
                if best_error is None or best_err_for_b < best_error:
                    best_error = best_err_for_b
                    best_pair = (best_a_for_b, b)
        
        if time.time() > deadline:
            break
    
    if best_error is None:
        return [11, 6]
    
    a, b = best_pair
    # Verify
    assert isinstance(a, int) and isinstance(b, int)
    assert 1 <= a <= MAX_N and 1 <= b <= MAX_N
    assert a != b
    assert abs(a / b - 1.5) > 0
    
    return [a, b]

# EVOLVE-BLOCK-END
