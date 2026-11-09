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

    # Self-inverse: 1 and 1008 (since 1^2 ≡ 1 and 1008^2 ≡ 1 mod 1009)
    # Interval must not include 1 or 1008
    lo_bound = 2
    hi_bound = 1007

    # Strategy: Model as finding the longest contiguous interval in [2, 1007]
    # that contains no pair (x, inv[x]) where both are in the interval.
    # 
    # Key insight: For each pair {a, b} where inv[a] = b (a != b), we cannot
    # have both a and b in our interval. Since we're looking for contiguous intervals,
    # we can think of this as: for each x, if inv[x] is "close" to x, it creates
    # a constraint on where we can place our interval.
    #
    # More precisely: An interval [L, R] is valid iff for all x in [L, R],
    # inv[x] < L or inv[x] > R.
    #
    # Structural approach: For each possible left endpoint L, find the maximum
    # right endpoint R such that [L, R] is valid. We can do this efficiently by
    # noting that as R increases, the constraint is: for all x in [L, R], inv[x] > R
    # (since inv[x] < L is already satisfied or not, and we need inv[x] not in [L,R]).
    #
    # Actually, let's use a cleaner structural approach:
    # For each pair (x, inv[x]) with x < inv[x], define the "forbidden span" as
    # the range [x, inv[x]] - any interval containing both endpoints is invalid.
    # But we need intervals that avoid ALL such pairs.
    #
    # Better structural view: Think of it as a constraint satisfaction.
    # For a fixed L, the maximum valid R is the minimum over all x in [L, ...]
    # of (inv[x] - 1) if inv[x] >= L, else infinity.
    # 
    # Let's compute for each L: max_R[L] = max R such that [L,R] is valid.
    # [L, R] is valid iff for all x in [L, R], not (L <= inv[x] <= R).
    # This is equivalent to: for all x in [L, R], inv[x] < L or inv[x] > R.
    # 
    # For fixed L, as R grows, the condition "inv[x] > R" becomes harder to satisfy
    # for x where inv[x] >= L. Specifically, if there exists x in [L, R] with
    # inv[x] in [L, R], it's invalid.
    #
    # Efficient computation: For each L, start with R = L, and extend R while valid.
    # The constraint when adding R: inv[R] must not be in [L, R], AND for all existing
    # x in [L, R-1], inv[x] must not equal R (i.e., inv[R] must not be in [L, R-1]).
    # Combined: inv[R] < L or inv[R] > R.
    #
    # But we also need to ensure that no x in [L, R-1] has inv[x] = R.
    # That happens iff inv[R] is in [L, R-1]. So the incremental check is:
    # inv[R] < L or inv[R] > R.
    #
    # This gives O(n^2) total, which is fine for n=1006.
    #
    # Let's also try a suffix-min approach for potential speedup:
    # For each position x, define constraint[x] = inv[x] if inv[x] >= x else -inf
    # (we only care about the "forward" constraint when building from left).
    # Actually, let's just do the direct O(n^2) with early termination.

    best_start = lo_bound
    best_len = 1

    for L in range(lo_bound, hi_bound + 1):
        # Find maximum R >= L such that [L, R] is valid
        max_R = L
        for R in range(L + 1, hi_bound + 1):
            # Check if adding R keeps [L, R] valid
            # Condition: inv[R] < L or inv[R] > R
            # (This ensures R's inverse is outside [L, R], and no x in [L,R-1] has inverse R)
            if not (inv[R] < L or inv[R] > R):
                break
            max_R = R

        length = max_R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

        # Deadline check
        if time.time() - start_time > DEADLINE:
            break

    # Final verification
    final_start = best_start
    final_end = best_start + best_len - 1
    valid = True
    for x in range(final_start, final_end + 1):
        if final_start <= inv[x] <= final_end:
            valid = False
            break

    if not valid:
        # Fallback: brute force search for any valid interval
        for L in range(lo_bound, hi_bound + 1):
            for R in range(hi_bound, L - 1, -1):
                ok = True
                for x in range(L, R + 1):
                    if L <= inv[x] <= R:
                        ok = False
                        break
                if ok:
                    best_start = L
                    best_len = R - L + 1
                    valid = True
                    break
            if valid:
                break

    return [best_start, best_len]
