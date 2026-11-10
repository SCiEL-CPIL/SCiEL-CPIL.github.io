import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # min_R_at_L[L] = min { R_x | L_x == L }
    min_R_at_L = [float('inf')] * (MOD + 2)
    
    for x in range(1, MOD):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
            
    # M[s] = min { R_x | L_x >= s } computed as suffix minimum of min_R_at_L
    M = [0] * (MOD + 2)
    curr_min = float('inf')
    for s in range(MOD - 1, 0, -1):
        if min_R_at_L[s] < curr_min:
            curr_min = min_R_at_L[s]
        M[s] = curr_min
        
    # Now find max length
    best_len = 0
    best_start = 1
    
    for s in range(1, MOD):
        # M[s] is min R_x for L_x >= s
        max_e = M[s] - 1
        if max_e > MOD - 1:
            max_e = MOD - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Verify the result is correct
    if best_len > 0:
        s = best_start
        e = s + best_len - 1
        valid = True
        for x in range(s, e + 1):
            for y in range(s, e + 1):
                if (x * y) % MOD == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback: brute force search
            best_len = 0
            best_start = 1
            for s2 in range(1, MOD):
                for e2 in range(s2, MOD):
                    ok = True
                    for x in range(s2, e2 + 1):
                        for y in range(s2, e2 + 1):
                            if (x * y) % MOD == 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        length = e2 - s2 + 1
                        if length > best_len:
                            best_len = length
                            best_start = s2
                    if time.time() > DEADLINE:
                        break
                if time.time() > DEADLINE:
                    break
    
    if best_len == 0:
        best_start = 2
        best_len = 1
    
    return [best_start, best_len]
