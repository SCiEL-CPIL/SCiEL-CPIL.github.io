"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    n = len(DIVISORS)
    divs = np.array(DIVISORS, dtype=np.int64)
    
    # We want to find integers x_i such that:
    # 1. x_i is divisible by divs[i]
    # 2. All x_i are distinct
    # 3. x_i <= MAX_VAL
    # 4. All x_i are positive
    # Minimize max(x_i) - min(x_i)
    
    # Key insight: if we fix a lower bound L, the optimal choice for each x_i
    # is the smallest multiple of divs[i] that is >= L, but we need them all distinct.
    # If two have the same value, we need to bump one up.
    
    # Better approach: try to find a window [L, L+W] such that for each divisor,
    # there exists a multiple in that window, and we can assign distinct multiples.
    
    # For a given L, the smallest multiple of d >= L is:
    #   ceil(L/d) * d
    # The next one is (ceil(L/d)+1) * d, etc.
    
    # Strategy: sweep L over a range, and for each L compute the "natural" assignment
    # (smallest multiple >= L for each divisor). If there are collisions, try to resolve
    # them by bumping some values up. Then compute the range.
    
    # To be efficient, we only need to consider L values that are multiples of some divisor,
    # or just below a multiple. Actually, the optimal L is likely to be a multiple of one
    # of the divisors, or just after one.
    
    # Let's use a smarter approach: 
    # For each divisor d, the multiples are d, 2d, 3d, ...
    # We want to pick one from each such that they're all in a tight window.
    
    # Alternative: binary search on the window width W. For a given W, check if there
    # exists a window [L, L+W] containing a distinct multiple of each divisor.
    # But checking existence is hard.
    
    # Let's go with the sweep approach but be smarter about which L to try.
    
    # The optimal L is likely near a multiple of the LCM or near where collisions happen.
    # Let's try L values that are multiples of each divisor, plus a few neighbors.
    
    # Actually, let's just do a direct sweep but limit it.
    # The range of L we need to check: the optimal window is small, so L can range
    # from 1 to MAX_VAL - 9 (since we need 10 distinct values).
    
    # But that's too many. Let's think differently.
    
    # For a given set of 10 distinct values in a window of width W, the window [min, max]
    # has width W. Each value x_i must be a multiple of divs[i].
    
    # Key observation: if we fix the minimum value m, then each x_i >= m and x_i is a
    # multiple of divs[i]. The smallest such x_i is ceil(m/divs[i]) * divs[i].
    # If all these are distinct and <= m + W, we have a valid solution with range <= W.
    
    # So for each m, compute the "greedy" assignment and check distinctness.
    # The optimal m is likely to be a multiple of one of the divisors.
    
    # Let's collect candidate m values: all multiples of each divisor up to MAX_VAL.
    # That's about sum(MAX_VAL/d) for d in DIVISORS ≈ 1000000*(1/6+1/10+1/14+...) 
    # ≈ 1000000 * 0.4 ≈ 400,000 candidates. For each, we do 10 operations. That's 4M ops.
    # Should be fast enough.
    
    # But we can do better: we don't need all multiples. We just need to check
    # where the "greedy" assignment changes, which is at multiples of each divisor.
    
    # Let's just do the sweep over all candidate m values (multiples of any divisor).
    
    # Generate candidate m values
    candidate_ms = set()
    for d in DIVISORS:
        for m in range(d, MAX_VAL + 1, d):
            candidate_ms.add(m)
    
    candidate_ms = sorted(candidate_ms)
    
    best_range = MAX_VAL
    best_set = None
    
    # Use numpy for speed
    divs_np = np.array(DIVISORS, dtype=np.int64)
    
    for m in candidate_ms:
        if time.time() > deadline:
            break
        
        # Compute smallest multiple of each divisor >= m
        # ceil(m/d) * d
        quotients = (m + divs_np - 1) // divs_np  # ceil division
        vals = quotients * divs_np
        
        # Check if all <= MAX_VAL
        if np.any(vals > MAX_VAL):
            continue
        
        # Check distinctness
        if len(np.unique(vals)) < n:
            # Try to resolve collisions by bumping up
            # For each collision, bump one of them up to the next multiple
            vals_list = vals.tolist()
            seen = set()
            valid = True
            for i in range(n):
                v = vals_list[i]
                while v in seen:
                    v += DIVISORS[i]
                    if v > MAX_VAL:
                        valid = False
                        break
                if not valid:
                    break
                seen.add(v)
                vals_list[i] = v
            
            if not valid:
                continue
            vals = np.array(vals_list, dtype=np.int64)
        
        r = int(np.max(vals) - np.min(vals))
        
        if r < best_range:
            best_range = r
            best_set = [int(v) for v in vals]
            
            # The minimum possible range for 10 distinct integers is 9
            if best_range <= 9:
                break
    
    if best_set is None:
        # Fallback: use the divisors themselves
        best_set = [int(d) for d in DIVISORS]
    
    return best_set
# EVOLVE-BLOCK-END
