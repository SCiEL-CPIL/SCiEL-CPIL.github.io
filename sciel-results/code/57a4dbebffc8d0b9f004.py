"""Optimized search for distinct multiples of divisor set."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    lcm_all = reduce(_lcm, divisors)
    
    # Precompute all non-empty subsets as (frozenset, size) for Hall's condition
    # Only need to check subsets where the union might be small
    subset_list = []
    for mask in range(1, 1 << n):
        bits = []
        for i in range(n):
            if mask & (1 << i):
                bits.append(i)
        subset_list.append((frozenset(bits), len(bits)))
    
    # Sort by size ascending - small sets fail Hall's condition first
    subset_list.sort(key=lambda x: x[1])
    
    best_val = float('inf')
    best_assign = None
    
    for W in range(9, 150):
        if time.time() > deadline:
            break
            
        # Precompute masks for this W
        # masks[i][r] is a bitmask of offsets m in [0, W] such that m % divisors[i] == r
        masks = []
        for i, d in enumerate(divisors):
            m_r = [0] * d
            for r in range(d):
                mask = 0
                start = r
                if start > W:
                    continue
                for m in range(start, W + 1, d):
                    mask |= (1 << m)
                m_r[r] = mask
            masks.append(m_r)
            
        # Iterate L
        for L in range(lcm_all):
            if time.time() > deadline:
                break
                
            # Get masks for this L
            current_masks = []
            for i, d in enumerate(divisors):
                r = (-L) % d
                current_masks.append(masks[i][r])
                
            # Check Hall's Condition - only check subsets that could potentially fail
            valid = True
            for bits_set, bits_size in subset_list:
                # Quick check: if any single element has no available offset
                if bits_size == 1:
                    idx = next(iter(bits_set))
                    if current_masks[idx] == 0:
                        valid = False
                        break
                else:
                    union_mask = 0
                    for idx in bits_set:
                        union_mask |= current_masks[idx]
                    if union_mask.bit_count() < bits_size:
                        valid = False
                        break
                    
            if valid:
                # Find assignment via backtracking with better ordering
                # Order by number of available offsets (ascending)
                order = sorted(range(n), key=lambda i: current_masks[i].bit_count())
                result = [None] * n
                used_offsets = 0  # Use bitmask instead of set for speed
                
                def backtrack(idx):
                    nonlocal used_offsets
                    if idx == len(order):
                        return True
                    i = order[idx]
                    mask = current_masks[i]
                    # Try each available offset
                    temp_mask = mask
                    while temp_mask:
                        # Get lowest set bit
                        bit = temp_mask & (-temp_mask)
                        m = bit.bit_length() - 1
                        if not (used_offsets & bit):
                            used_offsets |= bit
                            result[i] = L + m
                            if backtrack(idx + 1):
                                return True
                            used_offsets &= ~bit
                            result[i] = None
                        temp_mask &= ~bit
                    return False
                
                if backtrack(0):
                    val = max(result) - min(result)
                    if val < best_val:
                        best_val = val
                        best_assign = result[:]
                    return [int(x) for x in best_assign]
        
    if best_assign is None:
        # Fallback: try a simple construction
        best_assign = []
        for d in divisors:
            # Find a multiple that's likely to be in a tight range
            # Use a heuristic: start from some base
            found = False
            for base in range(1, 200):
                val = base * d
                if all(val != x for x in best_assign):
                    best_assign.append(val)
                    found = True
                    break
            if not found:
                # Just pick the next available
                for val in range(1, MAX_V + 1):
                    if val % d == 0 and all(val != x for x in best_assign):
                        best_assign.append(val)
                        break
    
    return [int(x) for x in best_assign]
