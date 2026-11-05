import time

def construct():
    deadline = time.time() + 85
    N = 1008
    MOD = 1009

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    best_len = 0
    best_start = 1
    
    left = 1
    window = set()
    
    for right in range(1, N + 1):
        if time.time() > deadline - 1:
            break
            
        # Check if current element is a self-inverse (forbidden diagonal)
        if right == 1 or right == 1008:
            # The window [left, right-1] is valid
            current_len = right - 1 - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left
            
            # Reset window for the next segment
            window.clear()
            left = right + 1
        else:
            # Check if the inverse of 'right' is already in the window
            # We need to move 'left' forward until the conflict is resolved
            while inv[right] in window:
                window.remove(left)
                left += 1
                
            # Add current element to window
            window.add(right)
            
            # Update best
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left

    return [best_start, best_len]
