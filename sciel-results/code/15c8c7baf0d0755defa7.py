import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    # Two-pointer sliding window to find the longest valid interval
    # An interval [L, R] is valid iff for all x in [L,R], inv[x] is not in [L,R]
    # Note: if inv[x] == x, then x is a self-inverse (1 or 1008), so x cannot be in the interval.
    best_len = 0
    best_start = 1
    L = 1

    for R in range(1, 1009):
        # Deadline guard (though O(N) is instant, good practice)
        if time.time() - start_time > 85:
            break
        
        # Check if adding R creates a conflict
        # Conflict if inv[R] is in [L, R]
        if L <= inv[R] <= R:
            # To resolve conflict, L must move past inv[R]
            # If inv[R] == R, L becomes R+1, effectively excluding R
            # If inv[R] < R, L becomes inv[R]+1, excluding the earlier conflict
            L = inv[R] + 1
        
        # Calculate current window length
        # If L > R, the window is empty, length 0
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L

    if best_len == 0:
        # Fallback, though theoretically there should always be a valid interval of length >= 1
        # e.g., [2,2] since 2*2=4 != 1
        return [2, 1]
    return [best_start, best_len]
