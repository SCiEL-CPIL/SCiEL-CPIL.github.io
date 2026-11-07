"""
Strategy: Sliding window over sorted list of all valid multiples.
1. Generate all valid multiples for each divisor up to 1,000,000.
2. Create a sorted list of (value, divisor_index) pairs.
3. Use two-pointer sliding window to find the shortest span where all 10 divisors
   are represented by distinct values.
4. Exploit periodicity: since the problem is periodic with period LCM, we only
   need to search within one period, but the sliding window naturally handles this.
"""

import time
from math import gcd
from functools import reduce
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit)
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Calculate LCM for periodicity understanding
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Generate all valid multiples for each divisor
    # For divisor d, multiples are d, 2d, 3d, ... up to max_val
    # We'll create a list of (value, divisor_idx) for all valid multiples
    all_multiples = []
    for i, d in enumerate(divisors):
        for k in range(1, max_val // d + 1):
            all_multiples.append((k * d, i))
    
    # Sort by value
    all_multiples.sort(key=lambda x: x[0])
    
    # Sliding window: find shortest span [L, R] where all n divisors are covered
    # with distinct values
    # We need to track: for each divisor, how many distinct values are in the window
    # and which specific values they are (to ensure distinctness)
    
    best_span = float('inf')
    best_left = 0
    best_right = 0
    best_assignments = None
    
    # Two pointers: left and right
    left = 0
    right = 0
    n = len(divisors)
    
    # For each divisor, track the set of values in the current window
    divisor_values = [set() for _ in range(n)]
    covered = 0  # number of divisors with at least one value in window
    
    while right < len(all_multiples):
        if time.time() > deadline:
            break
        
        # Add all_multiples[right] to window
        val, div_idx = all_multiples[right]
        if val not in divisor_values[div_idx]:
            if len(divisor_values[div_idx]) == 0:
                covered += 1
            divisor_values[div_idx].add(val)
        
        right += 1
        
        # Try to shrink from left while still having all divisors covered
        while covered == n and left < right:
            # Check if current span is better
            current_span = all_multiples[right - 1][0] - all_multiples[left][0]
            if current_span < best_span:
                best_span = current_span
                best_left = left
                best_right = right - 1
                # Store assignments: for each divisor, pick one value from its set
                # We need to ensure distinctness, but since we're using a set per divisor,
                # values within a divisor are distinct. But values across divisors might
                # conflict. We need to pick one value per divisor such that all are distinct.
                # For now, just record the window; we'll resolve conflicts later.
                best_assignments = [list(divisor_values[i]) for i in range(n)]
            
            # Remove all_multiples[left] from window
            val, div_idx = all_multiples[left]
            divisor_values[div_idx].discard(val)
            if len(divisor_values[div_idx]) == 0:
                covered -= 1
            
            left += 1
    
    # Now we need to resolve the best window: pick one distinct value per divisor
    if best_assignments is None:
        # Fallback
        return [int(d) for d in P['divisors']]
    
    # For the best window, we need to assign distinct values to divisors
    # Use backtracking with MRV (minimum remaining values)
    # Each divisor has a set of candidate values from the window
    # We need to pick one value per divisor, all distinct
    
    candidates = best_assignments
    n_div = len(divisors)
    
    # Order divisors by fewest candidates (MRV)
    order = sorted(range(n_div), key=lambda i: len(candidates[i]))
    
    used_values = set()
    assignment = [None] * n_div
    
    def backtrack(idx):
        if idx == n_div:
            return True
        div_idx = order[idx]
        for val in candidates[div_idx]:
            if val not in used_values:
                used_values.add(val)
                assignment[div_idx] = val
                if backtrack(idx + 1):
                    return True
                used_values.remove(val)
                assignment[div_idx] = None
        return False
    
    if not backtrack(0):
        # If backtracking fails, try a larger window
        # Search for next best windows
        # For now, fallback to seed
        return [int(d) for d in P['divisors']]
    
    # Validate
    result = []
    for i, val in enumerate(assignment):
        v = int(val)
        if not (1 <= v <= max_val) or v % divisors[i] != 0:
            return [int(d) for d in P['divisors']]
        result.append(v)
    
    if len(set(result)) != len(result):
        return [int(d) for d in P['divisors']]
    
    return result

# EVOLVE-BLOCK-END
