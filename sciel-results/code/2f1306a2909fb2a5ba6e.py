"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    import time
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # We need to evaluate n in [1000, 1000000], so we look at n+i for i in 1..24
    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    # lp[m] = smallest prime < 24 dividing m, or 0 if none
    lp = [0] * (MAX_VAL + 1)
    for p in PRIMES:
        for m in range(p, MAX_VAL + 1, p):
            if lp[m] == 0:
                lp[m] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    sp = [1] * (MAX_VAL + 1)
    for m in range(2, MAX_VAL + 1):
        if lp[m] == 0:
            sp[m] = 1
        else:
            sp[m] = sp[m // lp[m]] * lp[m]

    # Sliding window with frequency tracking for O(1) distinct count updates
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1, WINDOW + 1):
        v = sp[1000 + i]
        freq[v] += 1
        if freq[v] == 1:
            distinct += 1

    best_n = 1000
    best_distinct = distinct

    for n in range(1001, N + 1):
        if time.time() > deadline - 5:
            break
        # Remove sp[n] (was the leftmost element, index n+0 = n, but our window for n-1 was [n, n+23])
        # For n, window is [n+1, n+24]. Previous window for n-1 was [n, n+23].
        # So remove sp[n], add sp[n+24]
        v_remove = sp[n]
        freq[v_remove] -= 1
        if freq[v_remove] == 0:
            distinct -= 1

        v_add = sp[n + WINDOW]
        freq[v_add] += 1
        if freq[v_add] == 1:
            distinct += 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
