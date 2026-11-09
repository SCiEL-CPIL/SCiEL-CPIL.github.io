"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    MOD = 1009
    
    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 = 1 mod 1009 => x = 1 or x = 1008
    # These must be excluded from any valid interval.
    # So search in [2, 1007].
    
    lo_bound = 2
    hi_bound = 1007
    
    # For each start s, greedily extend e as far as possible.
    # [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # When extending from e to e+1, the new element is y = e+1.
    # It breaks validity iff inv[y] is in [s, e].
    
    best_len = 0
    best_start = lo_bound
    
    for s in range(lo_bound, hi_bound + 1):
        e = s
        while e < hi_bound:
            y = e + 1
            # Adding y breaks the interval if inv[y] is in [s, e]
            if s <= inv[y] <= e:
                break
            e += 1
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
