import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    P = 1009
    # Compute inverses
    inv = [0] * P
    for x in range(1, P):
        # x * inv[x] = 1 mod P
        # Since P is small, we can just loop or use pow
        inv[x] = pow(x, P - 2, P)
        
    # We want to find max length interval [s, e] in [1, 1008]
    # such that for all x in [s, e], inv[x] not in [s, e]
    
    best_start = 1
    best_len = 0
    
    # Optimization:
    # If s=1, inv[1]=1, so 1 is in interval -> invalid.
    # If e=1008, inv[1008]=1008, so 1008 is in interval -> invalid.
    # So s >= 2 and e <= 1007.
    
    # We can iterate s from 2 to 1007
    # For each s, find max e.
    # Condition: for all x in [s, e], s <= inv[x] <= e is False.
    # i.e., inv[x] < s or inv[x] > e.
    
    # Let's just do a nested loop with early termination.
    # P is 1009, so P^2 is ~10^6, which is fast.
    
    for s in range(2, 1008):
        # Check if s itself is valid start
        # We extend e as far as possible
        e = s
        valid = True
        while e < 1008:
            # Check if adding e+1 keeps it valid?
            # Actually, we need to check if the WHOLE interval [s, e] is valid.
            # If [s, e] is valid, is [s, e+1] valid?
            # It is valid if inv[e+1] not in [s, e+1] AND for all x in [s, e], inv[x] not in [s, e+1].
            # The second part means: for all x in [s, e], inv[x] != e+1.
            # i.e., (e+1) is not the inverse of any x in [s, e].
            # i.e., inv[e+1] not in [s, e].
            
            # So we can maintain a set of inverses of elements in [s, e].
            # Or simply, since we are extending, we just need to check:
            # 1. inv[e+1] not in [s, e+1]
            # 2. No x in [s, e] has inv[x] == e+1.
            
            # Condition 2 is equivalent to: inv[e+1] not in [s, e].
            # Because if inv[e+1] = y, then y is the unique x such that inv[x] = e+1.
            # So we just need to check if inv[e+1] is in [s, e].
            
            # Wait, is that sufficient?
            # If we have [s, e] valid.
            # Add e+1.
            # New interval [s, e+1].
            # Pairs to check:
            # - (e+1, e+1): inv[e+1] == e+1? (Only 1, 1008. e+1 <= 1007, so no).
            # - (e+1, x) for x in [s, e]: inv[e+1] == x? i.e., x == inv[e+1].
            #   So we need inv[e+1] not in [s, e].
            # - (x, y) for x, y in [s, e]: Already valid by induction.
            
            # So yes, we can extend if inv[e+1] < s or inv[e+1] > e+1.
            # Note: inv[e+1] > e+1 is same as inv[e+1] > e since inv[e+1] is integer.
            # And inv[e+1] cannot be e+1 (unless e+1 is 1 or 1008, which is excluded).
            
            next_e = e + 1
            if next_e > 1007:
                break
            
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break
        
        if e - s + 1 > best_len:
            best_len = e - s + 1
            best_start = s
            
        if time.time() > deadline:
            break
            
    return [best_start, best_len]
