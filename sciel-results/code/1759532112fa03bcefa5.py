"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Identify self-inverse elements: x^2 == 1 mod p
    self_inv = set()
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inv.add(x)
            
    best_start = 2
    best_length = 0
    
    # Sliding window / Two-pointer approach
    # L is the start of the current valid window
    # We iterate R from 1 to n
    # If adding R creates a conflict, we move L to resolve it.
    L = 1
    
    for R in range(1, n + 1):
        # Check if R itself is a self-inverse
        if R in self_inv:
            # The window cannot contain R.
            # Since we are adding R at the end, we must exclude R.
            # The next valid window starts after R.
            L = R + 1
        else:
            # Check if R's inverse is in the current window [L, R-1]
            j = inv[R]
            # If j is in [L, R-1], then the pair (j, R) is in the window, which is invalid.
            # We must move L to j + 1 to remove j from the window.
            if L <= j <= R - 1:
                L = j + 1
        
        # Now the window [L, R] is valid.
        # (If L > R, the window is empty, length 0).
        curr_length = R - L + 1
        if curr_length > best_length:
            best_length = curr_length
            best_start = L
            
    # Fallback if no valid interval found (should not happen, e.g., [2,2] is valid)
    if best_length == 0:
        return [2, 1]
        
    return [best_start, best_length]
# EVOLVE-BLOCK-END
