import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    best_start = 1
    best_len = 0

    left = 1

    for right in range(1, N + 1):
        if time.time() > deadline:
            break
            
        y = inv[right]
        
        # If y is in the current window [left, right], we must exclude it.
        # Since right is fixed, we must increase left past y.
        # If y == right, it's self-inverse, left becomes right + 1.
        # If y < right, and y >= left, left becomes y + 1.
        # If y < left or y > right, no change needed.
        
        if y <= right:
            if y >= left:
                left = y + 1
        
        if left <= right:
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left

    if best_len <= 0:
        # Fallback: find any valid single element
        for i in range(1, N + 1):
            if inv[i] != i:
                best_start = i
                best_len = 1
                break

    return [best_start, best_len]
