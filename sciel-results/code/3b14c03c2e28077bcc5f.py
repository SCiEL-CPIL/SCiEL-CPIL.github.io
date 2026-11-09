def construct():
    MOD = 1009
    N = 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    
    left = 1
    # Using a boolean array for O(1) membership check and update
    in_window = [False] * (MOD + 1)
    
    for right in range(1, N + 1):
        y = inv[right]
        
        if right == y:
            # Self-inverse: cannot be in any valid interval
            # Move left past right to exclude it
            # We must clear the window?
            # The window is [left, right]. If we set left = right + 1, the window is empty.
            # But we need to clear 'in_window' for elements < left?
            # Actually, if left jumps to right+1, all previous elements are < left, so they are out.
            # We should clear the flags for elements that were in the window?
            # It's easier to just maintain the window correctly.
            # If we set left = right + 1, the window is empty.
            # We should reset in_window for all? No, that's O(N).
            # However, self-inverses are rare (only 1 and 1008).
            # We can just clear the array or handle it carefully.
            # Since N is small (1008), clearing is fine.
            for i in range(1, N + 1):
                in_window[i] = False
            left = right + 1
            # Current length is 0
        else:
            # Check if inverse is already in the window
            if in_window[y]:
                # Conflict: y is in [left, right-1]
                # Move left until y is removed
                while left <= y:
                    in_window[left] = False
                    left += 1
            # Now y is not in the window (or was never there)
            # Add right to the window
            in_window[right] = True
            
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left
                
    return [best_start, best_len]
