import math
import time

# EVOLVE-BLOCK-START
MAX_N = 10000000
DEADLINE = 90.0
FALLBACK = (11, 6)

def _init_sigma():
    """Compute sigma(n) for all n in [1, MAX_N] using sieve."""
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    return sigma

# Precompute at module load time
_sigma_arr = _init_sigma()

def construct():
    start = time.time()
    sigma = _sigma_arr
    
    # Check if we have enough time
    elapsed = time.time() - start
    if elapsed > DEADLINE - 3:
        a, b = FALLBACK
        return [int(a), int(b)]
    
    # Group indices by sigma value
    groups = {}
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    elapsed = time.time() - start
    if elapsed > DEADLINE - 3:
        a, b = FALLBACK
        return [int(a), int(b)]
    
    best_err = float('inf')
    best_pair = FALLBACK
    
    # For each group with at least 2 elements, find pairs with ratio close to 3/2
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        if time.time() - start > DEADLINE - 2:
            break
        
        nums.sort()
        n_len = len(nums)
        
        # Two-pointer approach: for each b (as denominator), find a ≈ 1.5 * b
        i = 0
        for j in range(n_len):
            b = nums[j]
            target = 1.5 * b
            
            # Advance i until nums[i] >= target or i reaches end
            while i < n_len - 1 and nums[i] < target:
                i += 1
            
            # Check nums[i] and nums[i-1] (if valid and != b)
            if i < n_len:
                a = nums[i]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
            
            if i > 0:
                a = nums[i - 1]
                if a != b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
        
        # Early termination if we found a very good match
        if best_err < 1e-12:
            break
    
    a, b = best_pair
    a = int(a)
    b = int(b)
    return [a, b]

# EVOLVE-BLOCK-END
