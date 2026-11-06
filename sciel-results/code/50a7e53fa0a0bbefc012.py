import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MOD = 1009
    N = 1008

    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Identify self-inverse elements (x^2 % MOD == 1)
    self_inv = set()
    for i in range(1, N + 1):
        if i * i % MOD == 1:
            self_inv.add(i)

    best_start = 1
    best_len = 0
    L = 1

    for R in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # If R is a self-inverse element, it cannot be in any valid interval.
        # So we must move L past R.
        if R in self_inv:
            L = R + 1
            continue
        
        # Check if inv[R] is in the current window [L, R].
        # If inv[R] >= L, then inv[R] is in [L, R-1] (since inv[R] != R as R is not self-inv)
        # or inv[R] == R (impossible here).
        # Actually, if inv[R] >= L, it means inv[R] is in [L, R-1] or inv[R] == R.
        # Since R is not self-inverse, inv[R] != R.
        # If inv[R] >= L, then inv[R] is in [L, R-1].
        # This creates a pair (inv[R], R) in the interval.
        # To fix this, we must exclude inv[R] from the interval.
        # Since we are moving R forward, we move L past inv[R].
        if inv[R] >= L:
            L = inv[R] + 1
        
        # Now [L, R] is a valid candidate interval
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L

    # If no interval was found (should not happen, but for safety)
    if best_len == 0:
        # Find any single element that is not self-inverse
        for i in range(1, N + 1):
            if i not in self_inv:
                best_start = i
                best_len = 1
                break

    return [best_start, best_len]
