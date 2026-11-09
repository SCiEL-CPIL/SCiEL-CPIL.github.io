def construct():
    MOD = 1009
    N = 1008

    # Compute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # For each starting position s, find the maximum valid interval length
    # An interval [s, e] is valid if for all x in [s, e], inv[x] is not in [s, e]
    # (since x*inv[x] % MOD == 1, and we need x*y % MOD != 1 for all x,y in interval)
    # Note: x=1 and x=1008 are self-inverse, so they can't be in any valid interval.
    
    # For a given start s, the constraint is:
    # For each x in [s, e], inv[x] must not be in [s, e]
    # This means e < inv[x] or e > inv[x] is not enough - we need inv[x] < s or inv[x] > e
    
    # Let's think differently. For interval [s, e] to be valid:
    # For all x in [s, e]: inv[x] < s OR inv[x] > e
    # Equivalently: for all x in [s, e], NOT (s <= inv[x] <= e)
    
    # For a fixed start s, what's the maximum e?
    # e must be less than min{inv[x] : x in [s, e], s <= inv[x]} 
    # But this is circular. Let's use a different approach.
    
    # For each x, inv[x] is its constraint. If x is in the interval, then inv[x] must NOT be in the interval.
    # So if we have interval [s, e], for each x in [s, e], we need inv[x] < s or inv[x] > e.
    
    # Let's precompute: for each position i, what's the "blocking" inverse?
    # If we start at s, then for any x >= s, if inv[x] >= s, then we can't include both x and inv[x].
    
    # Better approach: for each start s, binary search or scan to find max e.
    # For a given s and e, valid iff for all x in [s,e], inv[x] < s or inv[x] > e.
    
    # Let's compute for each s the maximum valid e.
    # As e increases from s, we add constraint: for new x=e, need inv[e] < s or inv[e] > e (which is always true if inv[e] != e, i.e., e not self-inverse).
    # Wait, if inv[e] is in [s, e-1], then we have a conflict.
    
    # So for start s, as we extend e:
    # At each step, check if inv[e] is in [s, e-1]. If yes, we can't include e.
    # But we might be able to skip e and include later values? No, interval must be contiguous.
    
    # So for start s, the maximum valid interval is [s, e_max] where e_max is the largest e such that
    # for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # Let's compute this efficiently.
    # For each s from 1 to 1008:
    #   e = s
    #   while e <= 1008 and for all x in [s, e], inv[x] < s or inv[x] > e:
    #       e += 1
    #   valid_length = e - 1 - s + 1 if e > s else 0
    
    # But this is O(n^2) in worst case. n=1008, so 1008^2 ~ 1M, which is fine.
    
    best_start = 1
    best_len = 0
    
    for s in range(1, MOD):
        # Check if s itself is valid (s * s % MOD != 1)
        if (s * s) % MOD == 1:
            continue
        
        e = s
        valid = True
        while e < MOD:
            # Check if adding e to interval [s, e-1] maintains validity
            # Need: for all x in [s, e], inv[x] < s or inv[x] > e
            # We already know [s, e-1] is valid, so we just need to check x=e and also
            # that inv[e] doesn't conflict with existing [s, e-1]
            inv_e = inv[e]
            if s <= inv_e <= e:
                # Conflict: inv[e] is in [s, e], so we can't include e
                break
            # Also need to check that for all x in [s, e-1], inv[x] > e (since inv[x] < s was already checked)
            # Actually, the condition is: for all x in [s, e], inv[x] < s or inv[x] > e
            # For x in [s, e-1], we had inv[x] < s or inv[x] > e-1
            # Now we need inv[x] < s or inv[x] > e
            # If inv[x] was > e-1 but <= e, that's a problem. But inv[x] is integer, so inv[x] > e-1 means inv[x] >= e
            # So if inv[x] == e, that's a conflict.
            # But wait, we need to check all x in [s, e-1] for inv[x] == e?
            # That would be expensive. Let's think differently.
            
            # Alternative: maintain a set or check incrementally.
            # For interval [s, e] to be valid, we need:
            # 1. For all x in [s, e]: inv[x] < s or inv[x] > e
            # 2. No x in [s, e] has inv[x] in [s, e]
            
            # Condition 2 is equivalent to: for all x in [s, e], inv[x] not in [s, e]
            # Which is the same as condition 1.
            
            # The issue is checking condition 2 for all x in [s, e] at each step.
            # Let's precompute for each pair.
            
            # Actually, let's just do the O(n^2) check directly.
            e += 1
            # Now check if [s, e] is valid
            ok = True
            for x in range(s, e + 1):
                if s <= inv[x] <= e:
                    ok = False
                    break
            if not ok:
                e -= 1
                break
            # Continue to extend
        # After the loop, [s, e] is the maximum valid interval starting at s
        # Wait, the loop logic is a bit off. Let me redo.
        
        # Let me restructure: for start s, find max e such that [s,e] is valid.
        # Start with e = s. Check if valid. If yes, try e+1, etc.
        
        max_e = s
        # Check [s, s]
        if not (s <= inv[s] <= s):  # i.e., inv[s] != s
            max_e = s
            # Try to extend
            for e_candidate in range(s + 1, MOD):
                # Check if [s, e_candidate] is valid
                ok = True
                for x in range(s, e_candidate + 1):
                    if s <= inv[x] <= e_candidate:
                        ok = False
                        break
                if ok:
                    max_e = e_candidate
                else:
                    break
        
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    # Verify the result
    for x in range(best_start, best_start + best_len):
        for y in range(best_start, best_start + best_len):
            if (x * y) % MOD == 1:
                return [2, 1]
    
    return [best_start, best_len]
