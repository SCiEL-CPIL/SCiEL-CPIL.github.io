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

    best_len = 0
    best_start = 2

    # The interval must be contained in [2, 1007] because 1 and 1008 are self-inverse
    # 1*1 % 1009 == 1 and 1008*1008 % 1009 == 1
    # So any interval containing 1 or 1008 is invalid.
    
    # Use a two-pointer approach: for each right endpoint, find the earliest valid left endpoint
    # Or more efficiently: iterate start positions but use the fact that if [s, e] is valid,
    # then [s+1, e] is also valid (removing elements can't create new conflicts)
    
    # Better approach: for each possible end position, track what constraints it imposes
    # Actually, let's stick with the start-based approach but optimize the inner loop
    
    # Key optimization: if we know that extending from s to e fails because inv[e+1] is in [s,e],
    # then for s+1, we can start checking from a potentially different position
    
    # Let's use a smarter approach: precompute for each pair (x, inv[x]) the constraint
    # An interval [s, e] is valid iff for all x in [s,e], inv[x] is NOT in [s,e]
    
    # This means: for all x in [s,e], either inv[x] < s or inv[x] > e
    
    # Let's think differently: for a fixed e, what's the minimum s such that [s,e] is valid?
    # We need: for all x in [s,e], inv[x] < s or inv[x] > e
    # Equivalently: for all x in [s,e], NOT (s <= inv[x] <= e)
    # Which means: for all x in [s,e], inv[x] < s or inv[x] > e
    
    # For a fixed e, the constraint is: s > max{inv[x] : x in [s,e], inv[x] <= e} 
    # This is circular. Let me just use the original approach with minor optimizations.
    
    # Optimization 1: Use local variables for speed
    inv_local = inv
    LIMIT_local = LIMIT
    
    for s in range(2, 1008):
        e = s
        # Extend the interval to the right
        while e < 1007:
            next_val = e + 1
            next_inv = inv_local[next_val]
            # Conflict if next_inv is in the current interval [s, e]
            if next_inv >= s and next_inv <= e:
                break
            e = next_val

        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

        # Time check
        if time.time() - start_time > DEADLINE:
            break

    return [best_start, best_len]
