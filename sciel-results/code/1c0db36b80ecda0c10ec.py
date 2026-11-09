import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
    
    # Self-inverse elements: x*x % 1009 == 1 => x = 1 or x = 1008
    # Interval must be within [2, 1007]
    
    # Two-pointer approach: for each S, find max E such that [S,E] is valid.
    # Valid means: for all x in [S,E], inv[x] < S or inv[x] > E.
    # As S increases, E_max is non-decreasing (removing left elements only helps).
    
    best_len = 0
    best_start = 2
    
    S = 2
    E = 2
    
    while S <= 1007:
        if E < S:
            E = S
        
        # Extend E as far as possible
        while E + 1 <= 1007:
            nv = E + 1
            if S <= inv[nv] <= E:
                break
            E += 1
        
        current_len = E - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
        
        S += 1
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Final verification: confirm [best_start, best_start+best_len-1] is valid
    # and do a quick independent check to ensure optimality
    # Verify the found interval is valid
    end = best_start + best_len - 1
    valid = True
    for x in range(best_start, end + 1):
        ix = inv[x]
        if best_start <= ix <= end:
            valid = False
            break
    
    if not valid:
        # Fallback: brute force
        verified_best_len = 0
        verified_best_start = 2
        for s in range(2, 1008):
            e = s
            while e + 1 <= 1007:
                nv = e + 1
                if s <= inv[nv] <= e:
                    break
                e += 1
            cur_len = e - s + 1
            if cur_len > verified_best_len:
                verified_best_len = cur_len
                verified_best_start = s
        best_len = verified_best_len
        best_start = verified_best_start
    
    return [best_start, best_len]
