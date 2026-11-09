# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MOD = 1009
    N = 1008  # values are 1..1008

    # Compute modular inverse for each x in 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Self-inverse: x where inv[x] == x
    # For prime p, x^2 = 1 mod p => x = 1 or x = p-1
    # So 1 and 1008 are self-inverse, must be excluded
    lo_bound = 2
    hi_bound = 1007

    # Collect all forbidden pairs (a, b) with a < b, both in [2, 1007]
    # where b = inv[a]
    pairs = []
    seen = set()
    for x in range(lo_bound, hi_bound + 1):
        ix = inv[x]
        if ix == x:
            continue
        a = min(x, ix)
        b = max(x, ix)
        # Both must be in range
        if a < lo_bound or b > hi_bound:
            continue
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))

    # Sort pairs by a
    pairs.sort(key=lambda p: p[0])

    # For each start s, the constraint is: for every pair (a, b) with a >= s,
    # we must have e < b (i.e., b is not in the interval [s, e]).
    # So e_max(s) = min(b for all pairs (a,b) with a >= s) - 1
    # But also e <= hi_bound.

    # Precompute suffix minimum of b values
    INF = hi_bound + 1
    # min_b_at_a[a] = minimum b among pairs with that specific a
    min_b_at_a = [INF] * (hi_bound + 2)
    for (a, b) in pairs:
        if b < min_b_at_a[a]:
            min_b_at_a[a] = b

    # Suffix minimum: min_b_from[s] = min(min_b_at_a[s], min_b_at_a[s+1], ..., min_b_at_a[hi_bound])
    min_b_from = [INF] * (hi_bound + 2)
    min_b_from[hi_bound + 1] = INF
    for s in range(hi_bound, lo_bound - 1, -1):
        min_b_from[s] = min(min_b_at_a[s], min_b_from[s + 1])

    # Now find the best interval
    best_start = lo_bound
    best_len = 0

    for s in range(lo_bound, hi_bound + 1):
        max_e = min(min_b_from[s] - 1, hi_bound)
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Time check (should be fast, but safety)
    if time.time() - start_time > DEADLINE:
        pass

    return [best_start, best_len]
# EVOLVE-BLOCK-END
