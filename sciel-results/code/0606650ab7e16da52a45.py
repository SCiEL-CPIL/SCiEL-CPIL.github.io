import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s is plenty for N=1008

    MOD = 1009
    N = 1008

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
            
        x = right
        y = inv[x]

        if x == y:
            # Self-inverse, cannot be in any valid interval
            left = right + 1
            continue

        # If y is in the current window [left, right-1], we must exclude it
        if y >= left:
            left = y + 1

        # The window [left, right] is now valid
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left

    return [best_start, best_len]
