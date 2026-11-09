import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
        
    best_len = 0
    best_start = 1
    s = 1
    
    # Iterate through each potential end of the interval
    for e in range(1, MAX_VAL + 1):
        partner = inv[e]
        
        # Check for conflict with the new element e
        if partner == e:
            # Self-inverse: e cannot be in any valid interval
            s = e + 1
        else:
            # If the partner is currently in the window [s, e-1], 
            # we must move s past the partner.
            # Note: partner < e is implied if partner is in [s, e-1]
            if s <= partner < e:
                s = partner + 1
        
        # Calculate current valid interval length
        # If s > e, the interval is empty (length 0)
        current_len = e - s + 1
        
        if current_len > best_len:
            best_len = current_len
            best_start = s
            
        # Time check (not really needed for N=1008 but good practice per contract)
        if time.time() - start_time > 80:
            break
            
    # Verify the result is valid (defensive check)
    # Check that [best_start, best_start + best_len - 1] is valid
    end = best_start + best_len - 1
    for x in range(best_start, end + 1):
        for y in range(best_start, end + 1):
            if (x * y) % LIMIT == 1:
                # This shouldn't happen, but if it does, fall back to a known valid interval
                # Find a small valid interval
                for s2 in range(1, MAX_VAL + 1):
                    for l2 in range(1, MAX_VAL - s2 + 1):
                        e2 = s2 + l2 - 1
                        valid = True
                        for x2 in range(s2, e2 + 1):
                            for y2 in range(s2, e2 + 1):
                                if (x2 * y2) % LIMIT == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            return [s2, l2]
                return [1, 1]
    
    return [best_start, best_len]
