def construct():
    mod = 1009
    inv = [0] * 1009
    for x in range(1, 1009):
        for y in range(1, 1009):
            if (x * y) % mod == 1:
                inv[x] = y
                break
    
    best_start = 1
    best_len = 1
    
    for s in range(1, 1009):
        # Maximum valid e for start s
        # [s, e] is valid iff for all x in [s, e], inv[x] not in [s, e]
        # i.e., for all x in [s, e], inv[x] < s or inv[x] > e
        # The constraint is: e < min over x in [s, e] with inv[x] >= s of inv[x]... 
        # Actually, e must be < max(x, inv[x]) for any x where both x, inv[x] >= s and x in [s,e]
        # Let's find the smallest "blocking" pair
        
        max_e = 1008
        for x in range(s, 1009):
            if inv[x] >= s:
                # If we include x and inv[x] is also >= s, then e must be < max(x, inv[x])
                # But only if x is in [s, e], which it is if x <= e
                # The constraint is: if x <= e, then we need inv[x] > e (since inv[x] >= s)
                # So e < inv[x] if inv[x] >= s and x <= e
                # This means e must be < inv[x] for all x in [s, e] where inv[x] >= s
                # The tightest bound: e < min(inv[x] for x in [s, e] where inv[x] >= s)
                # But this is circular. Instead, note that the interval becomes invalid at
                # e = min over x where s <= x and s <= inv[x] of max(x, inv[x])
                # Because at e = max(x, inv[x]), both x and inv[x] are in [s, e]
                block = max(x, inv[x])
                if block < max_e:
                    max_e = block
                # We can break early if max_e < s (shouldn't happen)
        
        # The maximum valid e is max_e - 1, but at least s
        e_valid = max(s, max_e - 1)
        length = e_valid - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
