import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 1  # At minimum, any single element x where x*x != 1 mod 1009 is valid
    start_time = time.time()
    
    # For each start S in [2, 1007], greedily extend E as far as possible
    # The interval [S, E] is valid iff for all x in [S,E], inv(x) is NOT in [S,E]
    # Since we extend one at a time, we just check if inv(new_E) is in [S, current_E]
    
    for S in range(2, 1008):
        # Time guard with safety margin
        if time.time() - start_time > 50:
            break
        
        # Upper bound on possible length: can't exceed 1007 - S + 1
        # If even the max possible can't beat best_len, skip
        max_possible = 1007 - S + 1
        if max_possible <= best_len:
            break
        
        E = S - 1
        # Extend E as far as possible, up to 1007
        while E < 1007:
            new_E = E + 1
            inv_new_E = inv[new_E]
            
            # Check if adding new_E creates a conflict:
            # inv(new_E) must NOT be in [S, E]
            if S <= inv_new_E <= E:
                break
            E = new_E
        
        if E >= S:
            current_len = E - S + 1
            if current_len > best_len:
                best_len = current_len
                best_start = S
    
    return [best_start, best_len]
