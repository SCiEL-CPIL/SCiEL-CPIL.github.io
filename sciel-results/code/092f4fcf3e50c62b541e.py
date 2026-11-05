"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses: inv[x] = x^(-1) mod p for x in 1..1008
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # We need to find the longest interval [start, start+length-1] within [1, 1008]
    # such that for all x, y in the interval (including x=y), x*y % 1009 != 1.
    #
    # Key observations:
    # 1. Self-inverse elements: x*x % p == 1 means x^2 = 1 mod p, so x = 1 or x = p-1 = 1008.
    #    These cannot appear in any valid interval.
    # 2. For x != 1, x != 1008, the inverse inv[x] is distinct from x.
    #    So the constraint is: the interval must not contain both x and inv[x] for any x.
    #
    # Strategy: Sliding window / two-pointer approach.
    # We iterate the right endpoint R from 1 to 1008, maintaining the left endpoint L
    # such that [L, R] is the largest valid window ending at R.
    #
    # When we add R to the window, we check if inv[R] is already in [L, R-1].
    # If so, we must move L to inv[R] + 1 to exclude inv[R].
    # Also, if R itself is 1 or 1008, it's invalid, so we reset.
    
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        # Check if R is a self-inverse (1 or 1008)
        if R == 1 or R == 1008:
            # R cannot be in any valid interval
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        if L <= invR < R:
            L = invR + 1
        
        # Now [L, R] should be valid (by construction, we've resolved all conflicts)
        # But we need to double-check: could there be other conflicts?
        # Actually, the sliding window approach works because:
        # - Each pair {x, inv[x]} is disjoint (since inv is an involution)
        # - When we move L past invR, we only need to worry about pairs where
        #   one element is in [L, R] and the other is also in [L, R]
        # - The only new element added is R, and its only potential conflict is with invR
        # - Previous elements in [L, R-1] were already conflict-free among themselves
        
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Fallback: if no valid interval found (shouldn't happen), return a minimal one
    if best_len == 0:
        # Find any valid single element (not 1 or 1008)
        for x in range(2, 1008):
            if x * x % p != 1:
                return [x, 1]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
