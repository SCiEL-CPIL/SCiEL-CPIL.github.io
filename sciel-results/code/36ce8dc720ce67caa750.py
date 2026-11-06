def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Valid interval must be within [2, 1007] because 1 and 1008 are self-inverses
    # and cannot be in any valid interval (x*x % 1009 != 1 fails for x=1, 1008)
    
    left = 2
    best_start = 2
    best_len = 0
    
    # in_window[i] is True if i is in the current interval [left, right]
    in_window = [False] * (p + 1)
    
    for right in range(2, 1008): # 2 to 1007 inclusive
        inv_r = inv[right]
        
        # If inv_r is already in the window, we must move left until it's out
        # Note: inv_r cannot be right itself in this range [2, 1007]
        while in_window[inv_r]:
            in_window[left] = False
            left += 1
            
        # Add right to the window
        in_window[right] = True
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
            
    return [best_start, best_len]
