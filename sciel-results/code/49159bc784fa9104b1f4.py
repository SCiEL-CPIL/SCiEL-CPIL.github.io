import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 elements, indices 0 to 46

    # Split into two halves for meet-in-the-middle
    n1 = n // 2  # 23
    n2 = n - n1  # 24

    # Compute LCM of all denominators for exact integer arithmetic
    # LCM(2, 3, ..., 48)
    # We'll compute this once
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # L is a very large number, but Python handles big ints
    # L = lcm(2..48)
    # Let's compute it: it's 535422888000000000 (approx)
    # Actually let me compute more carefully in code

    # Precompute scaled values: v[i] = L // denominators[i]
    # Then sum of reciprocals = sum(v[i]) / L
    # We want sum(v[i]) < L (strictly)
    # Minimize L - sum(v[i])
    
    v = [L // d for d in denominators]
    
    # Generate all subset sums for part 1 (indices 0 to n1-1)
    # Using integer arithmetic for exactness
    sums1 = []
    for mask in range(1 << n1):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += v[idx]
            idx += 1
            m >>= 1
        sums1.append((s, mask))
        
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Generate all subset sums for part 2 (indices n1 to n-1)
    sums2 = []
    for mask in range(1 << n2):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += v[n1 + idx]
            idx += 1
            m >>= 1
        sums2.append((s, mask))
        
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Sort sums2 by sum
    sums2.sort(key=lambda x: x[0])
    sums2_values = [x[0] for x in sums2]
    
    best_deficit = L  # max possible deficit is L (empty set)
    best_mask1 = 0
    best_mask2 = 0
    
    import bisect
    
    # For each sum in part 1, find best match in part 2
    # We want s1 + s2 < L (strictly)
    # So s2 < L - s1
    # Find the rightmost s2 in sums2_values that is < L - s1
    
    for s1, m1 in sums1:
        target = L - s1  # s2 must be < target
        
        # Find the rightmost index in sums2_values that is < target
        # bisect_left returns the index where target would be inserted
        # All elements at index < pos are <= target? No, < target if we use bisect_left
        # bisect_left(sums2_values, target) gives the first index where sums2_values[i] >= target
        # So all elements at index < pos are < target
        # We want the largest such element, which is at index pos-1
        
        pos = bisect.bisect_left(sums2_values, target)
        
        # Check candidates at pos-1 and pos-2 (for safety)
        for check_idx in [pos - 1, pos - 2]:
            if 0 <= check_idx < len(sums2):
                s2, m2 = sums2[check_idx]
                total = s1 + s2
                if total < L:  # strictly less
                    deficit = L - total
                    if deficit < best_deficit:
                        best_deficit = deficit
                        best_mask1 = m1
                        best_mask2 = m2
    
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Convert best masks to indices
    selected = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected.append(n1 + i)
    
    # Verify with Fraction
    if not selected:
        # Edge case: empty set
        selected = [0]  # Just 1/2, deficit = 1/2
    
    total_frac = Fraction()
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
        
    if total_frac >= 1:
        # Should not happen with integer arithmetic, but just in case
        selected = [0, 1, 2, 3]
    
    return selected

def greedy_fallback(denominators):
    # Simple greedy: add largest available term if sum < 1
    selected = []
    total = Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
