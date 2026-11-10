"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008

    # Precompute modular inverses for 1 to 1008
    # inv[x] = x^(-1) mod 1009
    inv = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_length = 0
    
    # The interval must be within [2, 1007] because 1 and 1008 are self-inverse
    # (1*1=1, 1008*1008=1 mod 1009), so any interval containing 1 or 1008 is invalid.
    # Thus s >= 2 and e <= 1007.
    
    # For each start s, greedily extend e to the right.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e].
    # Greedy extension: when adding next_e = e+1, the only new constraint is that
    # inv[next_e] is not in [s, next_e]. Since inv is an involution, this is
    # equivalent to inv[next_e] not in [s, e] (as inv[next_e] != next_e for
    # next_e in [2, 1007]).
    
    for s in range(2, MAX_VAL):  # s from 2 to 1007
        if time.time() > deadline:
            break
            
        e = s
        
        # Try to extend the interval to the right
        while e < MAX_VAL - 1:  # e can go up to 1007
            next_e = e + 1
            # Check if adding next_e keeps the interval valid.
            # Condition: inv[next_e] must NOT be in [s, next_e]
            if not (s <= inv[next_e] <= next_e):
                e = next_e
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s

    # Fallback if no valid interval found (should not happen as [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
# EVOLVE-BLOCK-END
