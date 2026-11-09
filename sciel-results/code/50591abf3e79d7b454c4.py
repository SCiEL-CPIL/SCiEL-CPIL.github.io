def construct():
    p = 1009
    # Precompute modular inverses for 1..1008
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x^2 = 1 mod p => x = 1 or x = p-1
    # These cannot appear in any valid interval
    # For each position i in [2, 1007], determine if it can be in a valid interval
    # An interval [s, e] is valid iff for all x in [s,e]:
    #   - x != 1 and x != 1008 (diagonal check)
    #   - inv[x] not in [s, e] (off-diagonal check, since if inv[x] in [s,e] and inv[x] != x, we have a pair)
    # Note: if inv[x] == x, that's the diagonal case already handled
    
    # Build a "blocker" concept: for interval [s,e] to be valid,
    # for every x in [s,e], we need inv[x] < s or inv[x] > e
    # Equivalently, min(inv[x] for x in [s,e] where inv[x] != x) must be < s or... 
    # Actually simpler: for each s, find the maximum e such that [s,e] is valid.
    
    # For a fixed start s (s >= 2), the interval [s, e] is valid iff:
    #   1. e <= 1007 (since 1008 is self-inverse)
    #   2. For all x in [s, e], inv[x] < s or inv[x] > e
    #      i.e., no x in [s,e] has s <= inv[x] <= e
    
    # Efficient approach: for each s, the constraint is that e < min(inv[x] for x in [s, e] if inv[x] >= s)
    # This is circular. Better: for each s, the maximum valid e is the smallest value such that
    # some x in [s, e] has inv[x] in [s, e].
    
    # Alternative: precompute for each position i, the "constraint" it imposes.
    # If i is in the interval, then inv[i] must NOT be in the interval.
    # So if i is in [s,e], we need inv[i] < s or inv[i] > e.
    # This means: e < inv[i] (if inv[i] >= s) or s > inv[i] (if inv[i] < s, always true for s >= 2)
    
    # For a fixed s, the constraint from element i (where s <= i <= 1007) is:
    #   if inv[i] >= s, then e < inv[i]
    # So the maximum e for start s is: min(inv[i] for i in [s, 1007] where inv[i] >= s) - 1
    # But we also need e >= s, and e <= 1007.
    
    # Let's compute this efficiently.
    # For each s from 2 to 1007:
    #   max_e = 1007
    #   for i in range(s, 1008):
    #       if inv[i] >= s:
    #           max_e = min(max_e, inv[i] - 1)
    #   if max_e >= s:
    #       length = max_e - s + 1
    #       update best
    
    # This is O(n^2) worst case but n=1006, so ~1M operations, very fast.
    
    best_len = 0
    best_start = 2
    
    for s in range(2, 1008):  # s from 2 to 1007
        # Early termination: max possible length is 1007 - s + 1
        if 1007 - s + 1 <= best_len:
            break
        
        max_e = 1007
        for i in range(s, 1008):
            if inv[i] >= s:
                candidate = inv[i] - 1
                if candidate < max_e:
                    max_e = candidate
                    if max_e < s:
                        break
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    return [best_start, best_len]
