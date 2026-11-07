"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 10000000
    
    # Sieve for sigma: sigma[n] = sum of divisors of n
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        return [11, 6]
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        groups[s].append(n)
    
    # For each group, find pair (a, b) minimizing |a/b - 3/2| with a != b
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For a given b, optimal a is closest to 3b/2
    best_val = float('inf')
    best_pair = [11, 6]  # fallback
    
    # Only consider groups with at least 2 elements
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() - t0 > DEADLINE:
            break
        # Sort the list (it's already in order since we iterate n in order)
        # nums is sorted since we appended in increasing n order
        arr = nums  # already sorted
        ln = len(arr)
        
        # For each b in arr, find a in arr closest to 3b/2, a != b
        # Use binary search
        for i in range(ln):
            b = arr[i]
            target_a = 3.0 * b / 2.0
            # Binary search for target_a in arr
            # Find insertion point
            lo, hi = 0, ln
            while lo < hi:
                mid = (lo + hi) // 2
                if arr[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check candidates around lo
            for idx in (lo - 1, lo, lo + 1):
                if 0 <= idx < ln and idx != i:
                    a = arr[idx]
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| exactly
                    # = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue  # exact 3/2, invalid
                    den = 2 * b
                    val = num / den
                    if val < best_val:
                        best_val = val
                        best_pair = [a, b]
    
    # Also try reversed: for each a, find b closest to 2a/3
    # Actually the above covers all pairs since we iterate all b
    # But let's also check by iterating a as the "first" element
    # The above already checks all (a,b) pairs where we fix b and search a
    
    # Verify types
    a, b = best_pair
    if not isinstance(a, int) or not isinstance(b, int):
        a, b = int(a), int(b)
    if a <= 0 or b <= 0 or a > N or b > N:
        return [11, 6]
    # Verify sigma equality
    if sigma[a] != sigma[b]:
        return [11, 6]
    # Verify a != b (ratio not 1, and not exactly 3/2)
    if 2 * a == 3 * b:
        return [11, 6]
    
    return [a, b]
# EVOLVE-BLOCK-END
