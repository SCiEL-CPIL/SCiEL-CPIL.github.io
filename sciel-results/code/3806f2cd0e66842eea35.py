"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for all elements in [1, 1008]
    # inv[i] = i^(-1) mod 1009
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer approach to find the longest valid interval
    # L is the start of the current valid window
    # R is the end of the current window
    L = 1
    best_len = 0
    best_start = 1
    
    for R in range(1, N + 1):
        # 1. If R is a self-inverse element (1 or 1008), it cannot be in any valid interval.
        #    So we must start the next potential interval after R.
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        # 2. Ensure 1 is not in [L, R]
        #    If L <= 1 <= R, the interval contains 1, which is self-inverse.
        #    Since R > 1 (handled above), this only happens if L <= 1.
        if L <= 1 <= R:
            L = 2
            
        # 3. Ensure 1008 is not in [L, R]
        #    If L <= 1008 <= R, the interval contains 1008.
        #    Since R < 1008 (handled above for R=1008), this only happens if L <= 1008 and R >= 1008.
        #    But R max is 1008. If R=1008, we skipped. So R < 1008 here.
        #    Wait, if R < 1008, then 1008 is not in [L, R] because R < 1008.
        #    So this check is technically redundant if we handled R=1008, but good for safety.
        if L <= 1008 <= R:
            L = 1009

        # 4. Handle conflict with R's inverse
        #    If inv[R] is in [L, R-1], then both R and inv[R] are in the interval, which is invalid.
        #    To fix, we must move L to be greater than inv[R].
        if L <= inv[R] <= R - 1:
            L = inv[R] + 1
            
        # 5. Check if the current interval [L, R] is valid and non-empty
        #    It is possible that L > R after adjustments.
        if L <= R:
            current_len = R - L + 1
            if current_len > best_len:
                best_len = current_len
                best_start = L
                
        # Safety check for time
        if time.time() > deadline:
            break

    # Fallback if no valid interval found (should not happen, [2,2] is always valid)
    if best_len == 0:
        best_start = 2
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
