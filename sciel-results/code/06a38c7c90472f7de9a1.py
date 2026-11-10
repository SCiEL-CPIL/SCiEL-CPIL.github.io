"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008

    # Precompute modular inverses for 1 to 1008
    # inv[x] = x^(-1) mod 1009
    inv = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_length = 0
    
    # Two-pointer approach
    # s is the left boundary, e is the right boundary
    # e is monotonically non-decreasing as s increases
    e = 1
    for s in range(2, MAX_VAL):  # s from 2 to 1007
        if time.time() > deadline:
            break
            
        # Extend e as far as possible while [s, e] remains valid
        # Validity condition for adding next_e:
        # inv[next_e] must NOT be in [s, next_e]
        while e < MAX_VAL - 1:
            next_e = e + 1
            partner = inv[next_e]
            # If partner is in [s, next_e], we have a conflict
            if s <= partner <= next_e:
                break
            else:
                e = next_e
        
        # At this point, [s, e] is the longest valid interval starting at s
        # Note: e might be less than s if no valid interval exists, but [s,s] is always valid
        # because inv[s] != s for s in [2, 1007].
        if e >= s:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
        else:
            # This case shouldn't happen for s in [2, 1007] as [s,s] is valid
            # But if it does, we must ensure we don't return an invalid interval
            # Since [s,s] is valid, e should become at least s.
            # If e < s, it means the while loop didn't run or broke immediately?
            # If e < s, we should set e = s if valid.
            # Let's just force e to be at least s if [s,s] is valid.
            # We know [s,s] is valid.
            if inv[s] != s:
                e = s
                length = 1
                if length > best_length:
                    best_length = length
                    best_start = s

    # Fallback if no valid interval found (should not happen)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
# EVOLVE-BLOCK-END
