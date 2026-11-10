"""Find the longest interval [start, start+length-1] in [1,1008] where no two elements (including same element) are modular inverses mod 1009."""

import time

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin

    # Precompute modular inverses for 1 to N
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 1
    best_length = 0

    # Iterate over all possible start positions
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Try to extend the interval starting at s
        e = s - 1
        while e < N:
            next_e = e + 1
            # Check if adding next_e keeps the interval valid.
            # The new element is next_e. Its inverse is inv[next_e].
            # The condition for validity is that inv[next_e] is NOT in [s, next_e].
            # i.e., inv[next_e] < s or inv[next_e] > next_e.
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break

        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s

    return [best_start, best_length]
