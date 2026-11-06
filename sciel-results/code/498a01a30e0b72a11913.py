def construct():
    p = 1009
    n = 1008

    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Sliding window approach
    # For each right endpoint R, maintain the smallest left endpoint L such that
    # [L, R] contains no inverse pair (including self-inverse).
    # A window [L, R] is valid iff for every x in [L, R], inv[x] is not in [L, R].
    # Since inv is an involution (inv[inv[x]] = x), this is equivalent to:
    # for every x in [L, R], inv[x] < L or inv[x] > R.
    
    # We use a boolean array to track which elements are in the current window.
    in_window = [False] * (p + 1)
    
    best_start = 1
    best_length = 0
    
    L = 1  # Left endpoint of current window
    
    for R in range(1, n + 1):
        # If R is self-inverse, it cannot be in any valid window
        if inv[R] == R:
            # Clear the window
            for i in range(L, R):
                in_window[i] = False
            L = R + 1
            continue
        
        # Check if inv[R] is currently in the window [L, R-1]
        # If so, we need to move L forward until inv[R] is no longer in the window
        while L <= R - 1 and in_window[inv[R]]:
            in_window[L] = False
            L += 1
        
        # Now inv[R] is not in [L, R-1], so adding R keeps the window valid
        in_window[R] = True
        
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
    
    return [best_start, best_length]
