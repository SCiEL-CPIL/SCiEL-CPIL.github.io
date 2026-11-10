import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    best_start = 1
    best_length = 0
    
    # For each L, find the maximum R such that [L, R] is valid
    # [L, R] is valid iff for all x in [L, R], inv[x] is not in [L, R]
    # i.e., for all x in [L, R], inv[x] < L or inv[x] > R
    
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Extend R as far as possible
        # Start with R = L, check if [L, L] is valid
        # [L, L] is valid iff inv[L] < L or inv[L] > L, i.e., inv[L] != L
        # But also L*L % MOD != 1, which is the same condition
        
        # We'll track the minimum "exclusion" constraint
        # For the interval [L, R] to be valid, we need:
        #   For all x in [L, R]: inv[x] < L or inv[x] > R
        # 
        # As R increases, the constraint inv[x] > R becomes harder to satisfy.
        # The interval becomes invalid when there exists x in [L, R] with L <= inv[x] <= R.
        #
        # Let's find the smallest R >= L such that some x in [L, R] has inv[x] in [L, R].
        # For a given x, it causes a conflict at R = max(x, inv[x]) if L <= min(x, inv[x]).
        # So R_conflict = min{max(x, inv[x]) : x >= L, L <= min(x, inv[x])}
        #                = min{max(x, inv[x]) : x in [L, N], min(x, inv[x]) >= L}
        
        r_conflict = N + 1  # No conflict found means interval can extend to N
        
        for x in range(L, N + 1):
            if time.time() > deadline:
                break
            min_xy = min(x, inv[x])
            if min_xy >= L:
                # x is "active" for this L, and it creates a conflict at R = max(x, inv[x])
                max_xy = max(x, inv[x])
                if max_xy < r_conflict:
                    r_conflict = max_xy
        
        # Valid interval [L, R] has R < r_conflict, so max R = r_conflict - 1
        # But R must be >= L
        max_R = r_conflict - 1
        if max_R >= L:
            length = max_R - L + 1
            if length > best_length:
                best_length = length
                best_start = L
    
    if best_length == 0:
        # Fallback: return a trivial valid interval
        for x in range(2, N):  # skip 1 and 1008
            if x * x % MOD != 1:
                best_start = x
                best_length = 1
                break
        if best_length == 0:
            best_start = 2
            best_length = 1
    
    return [best_start, best_length]
