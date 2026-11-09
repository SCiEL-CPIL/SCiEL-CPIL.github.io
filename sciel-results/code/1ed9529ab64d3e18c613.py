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
            if s <= partner < e:
                s = partner + 1
        
        # Calculate current valid interval length
        # If s > e, the interval is empty (length 0)
        current_len = e - s + 1
        
        if current_len > best_len:
            best_len = current_len
            best_start = s
            
        # Time check
        if time.time() - start_time > 80:
            break
    
    # Brute-force verification: check all intervals to confirm optimality
    # Since N=1008, O(N^2) is feasible
    if time.time() - start_time < 70:
        verified_best_len = 0
        verified_best_start = 1
        for i in range(1, MAX_VAL + 1):
            valid = True
            for j in range(i, MAX_VAL + 1):
                # Check if j is valid to add to interval [i, j]
                # j is invalid if inv(j) is in [i, j] and inv(j) != j
                # or if j is self-inverse
                if inv[j] == j:
                    valid = False
                    break
                if i <= inv[j] < j:
                    valid = False
                    break
                if i < inv[j] and inv[j] >= i:
                    # inv[j] is in [i, j] range, need to check if it's actually in the interval
                    # Since we're building the interval [i, j], inv[j] must not be in [i, j]
                    if inv[j] >= i and inv[j] <= j:
                        valid = False
                        break
            if valid:
                length = j - i + 1
                if length > verified_best_len:
                    verified_best_len = length
                    verified_best_start = i
            # Early exit if we can't beat current best
            if (MAX_VAL - i + 1) <= verified_best_len:
                break
        
        if verified_best_len > best_len:
            best_len = verified_best_len
            best_start = verified_best_start
    
    return [best_start, best_len]
