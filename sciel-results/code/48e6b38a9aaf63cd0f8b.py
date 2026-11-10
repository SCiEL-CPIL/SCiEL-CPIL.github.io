import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each start s, find max valid end e
    best_start = 2
    best_len = 1
    
    for s in range(1, MOD):  # s from 1 to 1008
        if time.time() - start_time > DEADLINE:
            break
        # Check if s itself is self-inverse (1 or 1008)
        if inv[s] == s:
            continue  # s cannot be in any valid interval
        
        # Extend e from s upward
        max_e = s - 1
        for e in range(s, MOD):
            # Check if inv[e] is in [s, e]
            if s <= inv[e] <= e:
                break
            max_e = e
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    # Verify result
    s = best_start
    e = s + best_len - 1
    for x in range(s, e + 1):
        if s <= inv[x] <= e:
            # Should not happen
            break
    
    return [best_start, best_len]
