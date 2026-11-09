# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    MOD = 1009
    N = 1008
    
    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each x, define:
    #   lo[x] = min(x, inv[x])
    #   hi[x] = max(x, inv[x])
    # A pair {x, inv[x]} with x != inv[x] creates a conflict in [s, e] 
    # iff s <= lo[x] and hi[x] <= e
    
    # For fixed s, max valid e = min{hi[x] - 1 : x in 1..N, lo[x] >= s, x != inv[x]}
    # Bounded by N.
    
    # Precompute lo and hi for all x
    lo = [0] * (MOD + 1)
    hi = [0] * (MOD + 1)
    for x in range(1, MOD):
        if inv[x] == x:
            lo[x] = 0  # self-inverse, handled separately
            hi[x] = 0
        else:
            lo[x] = min(x, inv[x])
            hi[x] = max(x, inv[x])
    
    # For each possible lo value, find the min hi among pairs with that lo
    # Then for each s, we need min hi among pairs with lo >= s
    
    # min_hi_for_lo[l] = min{hi[x] : x, lo[x] == l, x != inv[x]}
    min_hi_for_lo = [MOD + 1] * (MOD + 1)
    for x in range(1, MOD):
        if inv[x] == x:
            continue
        l = lo[x]
        h = hi[x]
        if h < min_hi_for_lo[l]:
            min_hi_for_lo[l] = h
    
    # Suffix minimum: suff_min[s] = min{min_hi_for_lo[l] : l >= s}
    suff_min = [MOD + 1] * (MOD + 1)
    suff_min[MOD] = MOD + 1
    for i in range(MOD - 1, 0, -1):
        suff_min[i] = min(min_hi_for_lo[i], suff_min[i + 1])
    
    best_len = 0
    best_start = 1
    best_end = 1
    
    for s in range(1, N + 1):
        # Max valid end: suff_min[s] - 1, but also bounded by N
        # Also, s itself must not be a self-inverse
        if inv[s] == s:
            continue
        
        max_e = suff_min[s] - 1
        if max_e > N:
            max_e = N
        if max_e < s:
            continue
        
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            best_end = max_e
    
    # Verify
    s, e = best_start, best_end
    for x in range(s, e + 1):
        for y in range(s, e + 1):
            if (x * y) % MOD == 1:
                break
        else:
            continue
        break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
