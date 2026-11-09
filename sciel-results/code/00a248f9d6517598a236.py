import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    best_len = 0
    best_start = 2

    # The interval must be contained in [2, 1007] because 1 and 1008 are self-inverse
    # 1*1 % 1009 == 1 and 1008*1008 % 1009 == 1
    # So any interval containing 1 or 1008 is invalid.

    for s in range(2, 1008):
        # Extend the interval to the right
        e = s
        while e < 1007:
            next_val = e + 1
            next_inv = inv[next_val]
            # Conflict if next_inv is in the current interval [s, e]
            # If next_inv < s or next_inv > e, it is not in the interval, so no conflict
            if next_inv < s or next_inv > e:
                e = next_val
            else:
                # next_inv is in [s, e]
                # So adding next_val creates a pair (next_val, next_inv) with product 1
                break

        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s

        # Time check
        if time.time() - start_time > DEADLINE:
            break

    return [best_start, best_len]
