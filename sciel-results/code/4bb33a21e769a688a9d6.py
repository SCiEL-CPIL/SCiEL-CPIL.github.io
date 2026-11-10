def construct():
    MOD = 1009
    N = 1008  # elements are 1..1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)  # index 0..1008, we use 1..1008
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x where inv[x] == x
    # These are 1 and 1008
    forbidden = set()
    for x in range(1, MOD):
        if inv[x] == x:
            forbidden.add(x)
    
    # Two-pointer approach
    best_start = 1
    best_len = 0
    
    e = 0  # current right end (0-indexed, meaning empty window)
    # We'll iterate s from 1 to 1008
    for s in range(1, MOD):
        # Extend e as far as possible
        while e + 1 < MOD:
            candidate = e + 1
            # Check if candidate can be added
            if candidate in forbidden:
                break
            # Check if inv[candidate] is in [s, e]
            if inv[candidate] >= s and inv[candidate] <= e:
                break
            # Also need inv[candidate] != candidate (already handled by forbidden)
            e = candidate
        
        # Now window is [s, e]
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
        
        # Before moving s to s+1, if s == e, we need to reset
        # Actually, if s > e, we need to set e = s - 1
        if e < s:
            e = s - 1
    
    return [best_start, best_len]
