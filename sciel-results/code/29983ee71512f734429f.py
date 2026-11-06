"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # The problem: find the longest contiguous interval [s, e] within [1, 1008]
    # such that for ALL x, y in [s, e] (including x=y), x*y % 1009 != 1.
    # 
    # Key observations:
    # 1. Self-inverse elements (x where x*x % 1009 == 1) cannot be in the interval.
    #    These are x=1 and x=1008 (since 1008 = -1 mod 1009, and (-1)^2 = 1).
    # 2. For any x in the interval, inv[x] must NOT be in the interval.
    #
    # Approach: For each possible start s, find the maximum valid end e.
    # We can do this efficiently by noting that as we extend the interval,
    # the first conflict occurs when we include an element whose inverse
    # is already in the interval, or when we hit a self-inverse element.
    
    # Precompute: for each position i, what is the "conflict point"?
    # If we start at s, and extend to e, a conflict occurs at position e if:
    #   - e is self-inverse (1 or 1008), OR
    #   - inv[e] is in [s, e-1] (i.e., inv[e] >= s and inv[e] < e)
    #   - OR inv[e] > e and we will eventually reach inv[e]
    #
    # More precisely, for a fixed start s, the maximum end is:
    #   e_max = min(1007, min over all x in [s, e_max] of (inv[x]-1 if inv[x]>x else infinity))
    #   but also e_max < 1008 (can't include 1008) and e_max >= s.
    #
    # Actually, let's think differently. For a fixed start s:
    #   - We can include x in [s, e] only if inv[x] < s or inv[x] > e.
    #   - So for each x >= s, if inv[x] >= s, then we need e < inv[x].
    #   - The constraint is: e < min(inv[x] for all x in [s, e] where inv[x] >= s)
    #   - This is a self-referential constraint, so we solve it by scanning.
    
    best_start = 2
    best_len = 1  # [2, 2] is valid: 2*2 = 4 != 1 mod 1009
    
    for s in range(2, MOD - 1):  # s from 2 to 1007 (can't start at 1 or 1008)
        if time.time() > deadline:
            break
        
        # Find max valid end for start s
        # Scan from s upward, tracking the minimum "upper bound" imposed by inverses
        max_end = s
        for e in range(s, MOD):
            if time.time() > deadline:
                break
            if e >= 1008:
                break  # Can't include 1008
            ie = inv[e]
            if ie == e:
                break  # Self-inverse
            if ie < s:
                # No conflict: inverse is before the interval
                max_end = e
            elif ie < e:
                # Inverse is in [s, e-1], conflict!
                break
            else:
                # ie > e: inverse is ahead. We can include e, but must stop before ie.
                # So max_end is at most ie - 1, but we're at e now.
                # Continue, but remember the bound.
                max_end = e
                # We can continue as long as max_end < ie
                # But we need to check: will we hit ie? Only if we extend to ie.
                # For now, we can include e.
        
        L = max_end - s + 1
        if L > best_len:
            best_len = L
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
