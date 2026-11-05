import time

def construct():
    MOD = 1009
    N = 1008
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute modular inverses: inv[x] = x^(-1) mod 1009
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    t0 = time.time()

    # Two-pointer sliding window:
    # For each left endpoint L, find the maximal right endpoint R such that
    # for all x in [L, R], inv[x] is NOT in [L, R].
    #
    # Key invariant: when we advance R from R-1 to R, the only new constraint
    # is that inv[R] must not fall in [L, R-1] (since R itself is not in [L, R-1]).
    # When we advance L, we simply remove L from consideration; R never decreases.
    #
    # Self-inverse residues: 1 (1*1=1) and 1008 (1008*1008=1016064, 1016064%1009=1).
    # These cannot appear in any valid interval. Since L starts at 2 and R stops
    # before 1008, this is handled implicitly.

    best_len = 0
    best_start = 2

    L = 2
    R = 1  # window [L, R] is initially empty

    while L <= 1007:
        if time.time() - t0 > DEADLINE:
            break

        # Extend R as far as possible while maintaining validity
        while R < 1007:
            cand = R + 1
            # Adding cand is valid iff inv[cand] is NOT in [L, R]
            if L <= inv[cand] <= R:
                break
            R = cand

        # Record the maximal valid interval starting at L
        if R >= L:
            cur_len = R - L + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = L

        L += 1

    # Fallback: if no valid interval was found (shouldn't happen), return a trivial one
    if best_len == 0:
        return [2, 1]

    return [best_start, best_len]
