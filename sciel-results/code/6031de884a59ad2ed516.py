def construct():
    MOD = 1009
    N = MOD - 1
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    left = 1
    max_len = 0
    best_start = 1
    
    for right in range(1, N + 1):
        y = inv[right]
        
        # If right is self-inverse, it cannot be in any valid interval
        if y == right:
            left = right + 1
        else:
            # If the inverse of right is in the current window [left, right-1],
            # we must move left to exclude it.
            if y >= left:
                left = y + 1
        
        # The window [left, right] is now valid
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
            best_start = left
            
    return [best_start, max_len]
