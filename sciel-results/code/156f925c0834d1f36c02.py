"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each position i (1-indexed), compute the "conflict" constraint:
    # If i is in the interval, then inv[i] must NOT be in the interval.
    # Since the interval is contiguous [start, end], if i is in it, 
    # we need inv[i] < start or inv[i] > end.
    
    # Strategy: For each possible end position, find the maximum valid start.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # Equivalently: for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # For a fixed end e, the constraint is:
    # For all x in [s, e]: if inv[x] <= e then inv[x] < s, i.e., s > inv[x].
    # So s must be > max{inv[x] : x in [s,e] and inv[x] <= e}
    # But s appears on both sides. Let's think differently.
    
    # For a fixed end e, define:
    #   conflict_min[e] = min over x in [1..e] such that inv[x] <= e of inv[x]
    # Wait, that's not quite right either.
    
    # Let me think again. For interval [s, e]:
    # Valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    # 
    # For fixed e, as s increases, the condition becomes easier to satisfy
    # (fewer x in [s,e] to check, and s is larger so inv[x] < s is easier).
    # So for fixed e, there's a minimum valid s. Let's call it min_s[e].
    # 
    # min_s[e] = 1 + max{ inv[x] : x in [min_s[e], e] and inv[x] <= e }
    # This is circular. Let's use a different approach.
    #
    # Alternative: For each pair (i, j) where j = inv[i], if both i and j are in 
    # the interval, it's invalid. So the interval must not contain both i and inv[i]
    # for any i.
    #
    # For a contiguous interval [s, e], it's valid iff for no i do we have 
    # s <= i <= e AND s <= inv[i] <= e.
    #
    # Let's define for each position i, the "forbidden range" for the other endpoint.
    # If i is in [s, e], then inv[i] must be outside [s, e].
    # 
    # For fixed end e: we need s such that for all x in [s, e], inv[x] < s or inv[x] > e.
    # The binding constraint is: s > max{ inv[x] : x in [s, e] and inv[x] <= e }
    #
    # Let's compute this iteratively. For fixed e, start with s = 1 and increase 
    # until valid, or use a smarter approach.
    
    # Better approach: For each end e, compute the minimum s that works.
    # We can do a single pass. For each e from 1 to n:
    #   We maintain the constraint that s must be greater than certain values.
    
    # Actually, let's use this insight:
    # For interval [s, e] to be valid, we need:
    #   For all x in [s, e]: inv[x] < s or inv[x] > e
    # 
    # Define: for each e, let bad_s(e) be the set of s values that are invalid.
    # s is invalid for end e if there exists x in [s, e] with s <= inv[x] <= e.
    # Equivalently, s is invalid if there exists x with max(s, 1) <= x <= e and s <= inv[x] <= e.
    # 
    # For fixed e, s is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
    # 
    # Let's compute for each e the minimum valid s.
    # 
    # Key insight: For fixed e, if s is valid, then s+1 is also valid (removing 
    # element s from the interval can only help). So the valid s values form 
    # a suffix [min_s[e], e].
    
    # Compute min_s[e] for each e:
    # min_s[e] is the smallest s such that for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # Let's compute this efficiently.
    # For fixed e, consider all x in [1, e]. If inv[x] <= e, then x creates a 
    # constraint: if x >= s and inv[x] >= s, then s is invalid.
    # So s is invalid iff there exists x with x >= s and inv[x] >= s and inv[x] <= e and x <= e.
    # i.e., there exists x in [s, e] with inv[x] in [s, e].
    
    # For fixed e, define:
    #   f_e(s) = 1 if there exists x in [s, e] with inv[x] in [s, e], else 0
    # We want the smallest s with f_e(s) = 0.
    
    # Let's compute min_s[e] by starting from s=e and going down, or use a 
    # different method.
    
    # Alternative efficient method:
    # For each e, the constraint is that no pair (x, inv[x]) both lie in [s, e].
    # For each x with inv[x] <= e, the pair (x, inv[x]) is "active" and requires 
    # s > x or s > inv[x], i.e., s > max(x, inv[x]) is NOT required... 
    # Actually, to exclude both x and inv[x] from [s, e], we need s > x or s > inv[x] 
    # is wrong. We need NOT (s <= x <= e AND s <= inv[x] <= e).
    # Since we're fixing e and x <= e, inv[x] <= e, the condition for conflict is 
    # s <= min(x, inv[x]). So to avoid conflict, we need s > min(x, inv[x]).
    
    # So for fixed e, for each x in [1, e] with inv[x] <= e, we need s > min(x, inv[x]).
    # Therefore: min_s[e] = 1 + max{ min(x, inv[x]) : x in [1, e] and inv[x] <= e }
    # But we also need min_s[e] <= e (for nonempty interval).
    
    # Let's compute this:
    min_s = [0] * (n + 2)  # 1-indexed
    best_start = 1
    best_length = 0
    
    for e in range(1, n + 1):
        # Compute max of min(x, inv[x]) for x in [1, e] with inv[x] <= e
        max_conflict = 0
        for x in range(1, e + 1):
            ix = inv[x]
            if ix <= e:
                m = min(x, ix)
                if m > max_conflict:
                    max_conflict = m
        
        min_s[e] = max_conflict + 1
        
        if min_s[e] <= e:
            length = e - min_s[e] + 1
            if length > best_length:
                best_length = length
                best_start = min_s[e]
    
    # Verify the result
    s = best_start
    e = best_start + best_length - 1
    valid = True
    for x in range(s, e + 1):
        ix = inv[x]
        if s <= ix <= e:
            valid = False
            break
    
    if not valid:
        # Fallback to a known valid interval
        return [2, 30]
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
