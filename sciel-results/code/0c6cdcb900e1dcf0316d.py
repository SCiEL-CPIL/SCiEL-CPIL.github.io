import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest contiguous interval [s, s+l-1] subset of [1, 1008]
    such that for ALL x, y in the interval (including x=y), x*y % 1009 != 1.
    Maximize l.
    """
    p = 1009
    n = 1008
    deadline = time.time() + 55

    # 1. Precompute the inverse map
    inv_map = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = pow(x, p - 2, p)
        inv_map[ix] = x

    # 2. Sliding window
    # L is the start of the current valid window
    # R is the end of the current window
    L = 1
    best_len = 0
    best_start = 1

    for R in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Check if adding R creates a conflict
        # A conflict exists if inv[R] is in [L, R-1]
        # Or if R is a self-inverse (inv[R] == R), which means R conflicts with itself
        
        if R == 1 or R == 1008:
            # Self-inverse, cannot be in window
            L = R + 1
        else:
            if inv_map[R] >= L:
                # inv_map[R] is in [L, R-1] (since inv_map[R] != R and inv_map[R] <= R-1 is not guaranteed, 
                # but if inv_map[R] >= L, and we know inv_map[R] != R, 
                # we must check if inv_map[R] < R. 
                # If inv_map[R] > R, it's not in [L, R-1].
                # Wait, inv_map[R] is in [1, 1008].
                # If inv_map[R] > R, it is not in [L, R-1].
                # So we only have a conflict if L <= inv_map[R] < R.
                if inv_map[R] < R:
                    L = inv_map[R] + 1
                # else: inv_map[R] > R, no conflict with previous elements.
        
        # Now [L, R] is valid (or empty if L > R)
        if L <= R:
            curr_len = R - L + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = L

    return [best_start, best_len]
# EVOLVE-BLOCK-END
