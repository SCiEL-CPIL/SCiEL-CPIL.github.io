"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N_DIV = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute all valid multiples for each divisor
    # multiples[i] = sorted list of all multiples of DIVISORS[i] in [1, MAX_VAL]
    multiples = []
    for d in DIVISORS:
        mults = list(range(d, MAX_VAL + 1, d))
        multiples.append(mults)
    
    # Strategy: Sliding window over the sorted union of all multiples
    # Each element is (value, divisor_index)
    # We want to find the shortest window [L, R] in the sorted combined list
    # such that every divisor index 0..9 appears at least once in the window.
    
    # Build combined list of (value, div_idx)
    combined = []
    for i, mults in enumerate(multiples):
        for v in mults:
            combined.append((v, i))
    
    # Sort by value
    combined.sort(key=lambda x: x[0])
    
    # Sliding window with frequency counter
    best_span = float('inf')
    best_left_val = 0
    best_right_val = 0
    best_assignment = {}  # div_idx -> value
    
    freq = [0] * N_DIV
    left = 0
    n_combined = len(combined)
    
    for right in range(n_combined):
        val_r, div_r = combined[right]
        freq[div_r] += 1
        
        # Try to shrink from the left while all divisors are still covered
        while all(f > 0 for f in freq):
            val_l, div_l = combined[left]
            span = val_r - val_l
            if span < best_span:
                best_span = span
                best_left_val = val_l
                best_right_val = val_r
                # Record assignment: for each divisor, find one representative in [val_l, val_r]
                # We'll do this after finding the optimal window
                best_assignment = {}
                # Re-scan window to get one rep per divisor
                for i in range(left, right + 1):
                    v, d = combined[i]
                    if d not in best_assignment:
                        best_assignment[d] = v
            
            # Check if we can move left
            if freq[div_l] > 1:
                freq[div_l] -= 1
                left += 1
            else:
                break
        
        # Time check
        if time.time() > deadline:
            break
    
    # If we found a best window, build the final assignment
    if best_assignment and len(best_assignment) == N_DIV:
        result = [best_assignment[i] for i in range(N_DIV)]
        # Verify distinctness - if not distinct, need to fix
        if len(set(result)) == N_DIV:
            return [int(x) for x in result]
    
    # Fallback: greedy assignment
    # For each divisor, pick a multiple close to the center of the best window
    center = (best_left_val + best_right_val) // 2 if best_span < float('inf') else MAX_VAL // 2
    
    used = set()
    result = []
    for i, d in enumerate(DIVISORS):
        # Find multiple of d closest to center, not already used
        k = max(1, center // d)
        candidates = []
        for offset in range(0, 200):
            for kk in [k - offset, k + offset]:
                if kk >= 1:
                    v = kk * d
                    if v <= MAX_VAL and v not in used:
                        candidates.append(v)
            if len(candidates) > 50:
                break
        if candidates:
            # Pick the one closest to center
            best_v = min(candidates, key=lambda v: abs(v - center))
            used.add(best_v)
            result.append(best_v)
        else:
            # Just pick any unused multiple
            for v in multiples[i]:
                if v not in used:
                    used.add(v)
                    result.append(v)
                    break
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
