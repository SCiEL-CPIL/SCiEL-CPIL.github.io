"""Rewritten program: meet-in-the-middle exhaustive search for large reciprocal-subsum-free sets."""
import time
from fractions import Fraction
from itertools import combinations

def _is_valid_subset(subset):
    """Check if subset is reciprocal-subsum-free.
    For each element x in subset, 1/x must NOT equal sum of 1/y for any nonempty subset of subset \\ {x}.
    """
    n = len(subset)
    recip = {v: Fraction(1, v) for v in subset}
    # Precompute all subset sums of reciprocals for each "other" set
    # For efficiency, for each target x, we need to check if 1/x is in the set of subset-sums of {1/y : y in subset, y != x}
    # Use meet-in-the-middle for the subset-sum check when n-1 is large
    for x in subset:
        target = recip[x]
        others = [v for v in subset if v != x]
        m = len(others)
        if m == 0:
            continue
        # Meet in the middle
        half = m // 2
        left = others[:half]
        right = others[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = Fraction(0, 1)
            for i in range(len(left)):
                if mask & (1 << i):
                    s += recip[left[i]]
            left_sums.add(s)
        
        # For each subset sum of right, check if target - that sum is in left_sums
        for mask in range(1 << len(right)):
            s = Fraction(0, 1)
            for i in range(len(right)):
                if mask & (1 << i):
                    s += recip[right[i]]
            if (target - s) in left_sums:
                # Found a subset of others whose reciprocals sum to 1/x
                return False
    return True

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Start with the known good set of size 12
    best_set = list(range(13, 25))  # [13,14,...,24]
    
    # Try to find larger sets by exhaustive search
    # Search sizes from 16 down to 13, and also try to improve within size 12
    # Actually, let's search from larger sizes down
    
    for size in range(17, 11, -1):
        if time.time() > deadline:
            break
        
        # Try all combinations of size 'size' from [1..24]
        # This is C(24, size) which can be large
        # C(24,16) = 134596, C(24,15) = 42504, C(24,14) = 10626, C(24,13) = 2496, C(24,12) = 270416
        # For size 16: 134596 combos, each check is expensive
        # Let's be smart: only try if size > len(best_set)
        if size <= len(best_set):
            continue
            
        found = False
        for combo in combinations(range(1, 25), size):
            if time.time() > deadline:
                break
            if _is_valid_subset(combo):
                best_set = list(combo)
                found = True
                break
        if found:
            # Found a larger set, continue to try even larger? No, we're going down.
            # Actually we're iterating from 17 down, so if we find size 17, we should stop since we want max.
            # But wait, we're going from 17 down to 12. If we find size 17, that's the max we're checking.
            # Let's restructure: check from high to low, return first valid found.
            break
    
    # If no larger set found, try to verify/improve the current best
    # The seed returns [13..24] of size 12. Let's make sure it's valid.
    if not _is_valid_subset(best_set):
        # Fallback to a known valid small set
        best_set = list(range(13, 25))
    
    return best_set
