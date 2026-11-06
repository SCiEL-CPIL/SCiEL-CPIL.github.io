"""Rewritten program for closest strict underpacking of unit fractions."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Use Fraction for exact arithmetic
    # We want to maximize sum(1/d_i) < 1
    
    # Meet-in-the-middle with exact fractions
    # Split into two halves
    mid = n // 2
    
    def gen_sums_exactly(denom_list):
        """Generate all subset sums as Fractions with their masks."""
        sums = [(Fraction(0), 0)]
        for i, d in enumerate(denom_list):
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                new_s = s + Fraction(1, d)
                # Only keep if strictly less than 1
                if new_s < 1:
                    sums.append((new_s, m | (1 << i)))
        return sums
    
    deadline = time.time() + 55
    
    # Generate sums for first half
    half1 = denominators[:mid]
    half2 = denominators[mid:]
    
    sums1 = gen_sums_exactly(half1)
    sums2 = gen_sums_exactly(half2)
    
    # For half2, we want to quickly find the best sum2 such that sum1 + sum2 < 1
    # Sort sums2 by value
    sums2.sort(key=lambda x: x[0])
    
    # Extract just the values for binary search
    values2 = [s[0] for s in sums2]
    
    best_deficit = Fraction(1)
    best_mask1 = 0
    best_mask2 = 0
    
    # For each sum in half1, find the best sum in half2
    # We want max sum2 such that sum2 < 1 - sum1
    from bisect import bisect_left
    
    for s1, m1 in sums1:
        if time.time() > deadline - 2:
            break
        
        target = Fraction(1) - s1
        if target <= 0:
            continue
            
        # Find the largest sum2 strictly less than target
        # bisect_left gives the first index where value >= target
        idx = bisect_left(values2, target)
        
        if idx > 0:
            s2 = values2[idx - 1]
            total = s1 + s2
            deficit = Fraction(1) - total
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = m1
                best_mask2 = sums2[idx - 1][1]
                
                # If deficit is very small, we can early exit
                # The smallest possible positive deficit with these denominators
                # is 1/LCM, but we can stop if we find something very small
                if best_deficit < Fraction(1, 1000000):
                    break
    
    # Reconstruct indices
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)
            
    for i in range(len(half2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)
    
    # If no valid subset found, return a single element
    if not final_indices:
        # Return index 0 (denominator 2)
        final_indices = [0]
        
    return final_indices
# EVOLVE-BLOCK-END
