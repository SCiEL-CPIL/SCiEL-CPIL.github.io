import time
from math import gcd
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin before 60s kill
    
    denominators = list(range(2, 49))  # 47 denominators, indices 0..46
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2  # 23
    left = list(range(mid))      # indices 0..22
    right = list(range(mid, n))  # indices 23..46
    
    # Generate all subset sums for left half
    # Use float for speed, verify with Fraction later
    left_sums = [(0.0, 0)]  # (sum, bitmask)
    for i in left:
        val = 1.0 / denominators[i]
        new_entries = []
        for s, mask in left_sums:
            new_s = s + val
            new_entries.append((new_s, mask | (1 << i)))
        left_sums.extend(new_entries)
    
    # Generate all subset sums for right half
    right_sums = [(0.0, 0)]
    for i in right:
        val = 1.0 / denominators[i]
        new_entries = []
        for s, mask in right_sums:
            new_s = s + val
            new_entries.append((new_s, mask | (1 << i)))
        right_sums.extend(new_entries)
    
    # Sort right_sums by sum
    right_sums.sort(key=lambda x: x[0])
    right_vals = [s for s, _ in right_sums]
    
    best_deficit = float('inf')
    best_left_mask = 0
    best_right_mask = 0
    best_sum = 0.0
    
    # For each left sum, find best right sum
    for ls, lm in left_sums:
        if ls >= 1.0:
            continue
        # Need rs < 1.0 - ls
        remaining = 1.0 - ls
        # Find largest rs < remaining
        idx = bisect.bisect_left(right_vals, remaining)
        if idx > 0:
            rs, rm = right_sums[idx - 1]
            total = ls + rs
            if total < 1.0:
                deficit = 1.0 - total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_mask = lm
                    best_right_mask = rm
                    best_sum = total
    
    # Convert masks to indices
    result = []
    for i in left:
        if best_left_mask & (1 << i):
            result.append(i)
    for i in right:
        if best_right_mask & (1 << i):
            result.append(i)
    
    # Verify with Fraction
    frac_sum = Fraction(0)
    for idx in result:
        frac_sum += Fraction(1, denominators[idx])
    
    if frac_sum >= 1:
        # Shouldn't happen due to float precision, but fallback
        # Remove the smallest element
        if result:
            result.pop()
            frac_sum -= Fraction(1, denominators[result[-1]]) if result else 0
            # Recompute properly
            frac_sum = Fraction(0)
            for idx in result:
                frac_sum += Fraction(1, denominators[idx])
    
    if not result:
        return [0]  # Just 1/2
    
    return sorted(result)

def greedy_fallback(denominators, L, target):
    """Simple greedy fallback."""
    selected = []
    current_sum = Fraction(0)
    
    for d in sorted(denominators):
        w = Fraction(1, d)
        if current_sum + w < 1:
            selected.append(d)
            current_sum += w
    
    result = [denominators.index(d) for d in selected]
    return result
