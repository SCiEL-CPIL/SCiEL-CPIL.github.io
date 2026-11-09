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
    # For each start s, we need to find the maximum e such that no pair (a,b)
    # has both a and b in [s, e]. Since a < b, this means: for all pairs with
    # a >= s, we need b > e OR a > e.
    # The constraining pairs are those with a >= s and a <= e and b <= e.
    # We use an iterative approach: start with e = hi_bound, then narrow down.
    best_start = lo_bound
    best_len = 0

    for s in range(lo_bound, hi_bound + 1):
        # Iteratively find max valid end
        e = hi_bound
        for _ in range(20):  # should converge quickly
            # Find min b among pairs with a >= s and a <= e
            min_b = hi_bound + 1
            for (a, b) in pairs:
                if a >= s and a <= e:
                    if b < min_b:
                        min_b = b
            new_e = min(min_b - 1, hi_bound)
            if new_e >= e:
                break
            e = new_e
            if e < s:
                break
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    # Time check (should be fast, but safety)
    if time.time() - start_time > DEADLINE:
        pass

    # Verify the solution is valid
    # For every x, y in [best_start, best_start+best_len-1], check x*y % 1009 != 1
    end = best_start + best_len - 1
    valid = True
    for x in range(best_start, end + 1):
        for y in range(best_start, end + 1):
            if (x * y) % MOD == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: find a valid interval by brute force
        for s in range(lo_bound, hi_bound + 1):
            for e in range(hi_bound, s - 1, -1):
                ok = True
                for x in range(s, e + 1):
                    if (x * x) % MOD == 1:
                        ok = False
                        break
                    if inv[x] >= s and inv[x] <= e:
                        ok = False
                        break
                if ok:
                    return [s, e - s + 1]

    return [best_start, best_len]
# EVOLVE-BLOCK-END
