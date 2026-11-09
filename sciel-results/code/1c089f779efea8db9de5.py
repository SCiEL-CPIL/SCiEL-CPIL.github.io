import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        try:
            inv[i] = pow(i, -1, LIMIT)
        except TypeError:
            inv[i] = pow(i, LIMIT - 2, LIMIT)
            
    best_len = 0
    best_start = 2
    
    # Two-pointer sliding window
    # We want to find the longest interval [L, R] such that no pair (x, y) in it satisfies x*y == 1 mod 1009.
    # This is equivalent to: for all x in [L, R], inv[x] is not in [L, R].
    
    in_interval = [False] * (LIMIT + 1)
    L = 2
    R = 2
    
    # We iterate L from 2 to 1007
    # R is non-decreasing
    while L <= 1007:
        # Extend R as far as possible
        while R <= 1007:
            # Check if adding R creates a conflict
            # Conflict if inv[R] is already in the interval [L, R-1]
            if in_interval[inv[R]]:
                break
            # No conflict, add R to interval
            in_interval[R] = True
            R += 1
            
        # Now [L, R-1] is the longest valid interval starting at L
        current_len = (R - 1) - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
            
        # Remove L from interval before moving to next L
        in_interval[L] = False
        L += 1
        
        # Time check (should not trigger, but safe)
        if time.time() - start_time > 80:
            break
            
    return [best_start, best_len]
