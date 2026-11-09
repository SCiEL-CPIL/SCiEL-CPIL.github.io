"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import bisect
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    n = 30
    half = 15
    
    # First half: indices 0..14 correspond to 1/1, 1/2, ..., 1/15
    # Second half: indices 15..29 correspond to 1/16, 1/17, ..., 1/30
    
    # Enumerate all 3^15 sums for the first half using Fraction
    # Each coefficient is in {-1, 0, 1}
    
    def enumerate_sums(indices):
        """Enumerate all sums of sum(c_i / (i+1)) for i in indices, c_i in {-1,0,1}."""
        m = len(indices)
        results = [Fraction(0, 1)]
        for i in indices:
            denom = i + 1  # 1/(i+1)
            new_results = []
            for s in results:
                new_results.append(s)           # c = 0
                new_results.append(s + Fraction(1, denom))   # c = 1
                new_results.append(s - Fraction(1, denom))   # c = -1
            results = new_results
        return results
    
    first_indices = list(range(half))       # 0..14 -> 1/1..1/15
    second_indices = list(range(half, n))   # 15..29 -> 1/16..1/30
    
    # Enumerate first half sums
    sums1 = enumerate_sums(first_indices)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return seed answer
        return [0] * 28 + [1, -1]
    
    # Sort sums1 for binary search
    sums1_sorted = sorted(sums1)
    
    # Enumerate second half sums
    sums2 = enumerate_sums(second_indices)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # For each s2 in sums2, find s1 in sums1_sorted that minimizes |s1 + s2| with s1+s2 != 0
    best_val = None
    best_s1_idx = -1
    best_s2_idx = -1
    
    for i2, s2 in enumerate(sums2):
        # We want s1 such that s1 + s2 is as close to 0 as possible but not 0
        # So we want s1 close to -s2
        target = -s2
        
        # Binary search for target in sums1_sorted
        pos = bisect.bisect_left(sums1_sorted, target)
        
        # Check candidates around pos
        candidates = []
        for p in [pos - 1, pos, pos + 1]:
            if 0 <= p < len(sums1_sorted):
                candidates.append(p)
        
        for p in candidates:
            s1 = sums1_sorted[p]
            total = s1 + s2
            if total == 0:
                continue
            if best_val is None or abs(total) < abs(best_val):
                best_val = total
                best_s1_idx = p
                best_s2_idx = i2
        
        if time.time() > deadline:
            break
    
    if best_val is None:
        return [0] * 28 + [1, -1]
    
    # Recover coefficients
    # For s1: we need to recover which coefficients gave sums1_sorted[best_s1_idx]
    # This is tricky because we lost the mapping. Let me re-enumerate with tracking.
    
    # Better approach: re-enumerate with coefficient tracking
    def enumerate_sums_tracked(indices):
        """Return list of (sum, coeffs_tuple) for all combinations."""
        m = len(indices)
        results = [(Fraction(0, 1), (0,) * m)]
        for i in indices:
            denom = i + 1
            new_results = []
            for s, coeffs in results:
                new_results.append((s, coeffs + (0,)))
                new_results.append((s + Fraction(1, denom), coeffs + (1,)))
                new_results.append((s - Fraction(1, denom), coeffs + (-1,)))
            results = new_results
        return results
    
    if time.time() < deadline - 10:
        # Re-enumerate with tracking to recover coefficients
        sums1_tracked = enumerate_sums_tracked(first_indices)
        sums2_tracked = enumerate_sums_tracked(second_indices)
        
        # Sort sums1_tracked by sum value
        sums1_tracked_sorted = sorted(sums1_tracked, key=lambda x: x[0])
        
        best_val2 = None
        best_coeffs = None
        
        for i2, (s2, coeffs2) in enumerate(sums2_tracked):
            target = -s2
            pos = bisect.bisect_left([x[0] for x in sums1_tracked_sorted], target)
            
            candidates = []
            for p in [pos - 1, pos, pos + 1]:
                if 0 <= p < len(sums1_tracked_sorted):
                    candidates.append(p)
            
            for p in candidates:
                s1, coeffs1 = sums1_tracked_sorted[p]
                total = s1 + s2
                if total == 0:
                    continue
                if best_val2 is None or abs(total) < abs(best_val2):
                    best_val2 = total
                    best_coeffs = coeffs1 + coeffs2
            
            if time.time() > deadline - 5:
                break
        
        if best_coeffs is not None:
            return list(best_coeffs)
    
    # Fallback: reconstruct from best_s1_idx by re-enumerating
    # This is complex; use the seed fallback
    return [0] * 28 + [1, -1]

# EVOLVE-BLOCK-END
