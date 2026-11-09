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

    # Max value needed is 1000000 + 24 = 1000024
    MAX_VAL = N + WINDOW

    # Compute smallest prime factor restricted to primes < 24
    # lp[m] = smallest prime < 24 dividing m, or 0 if none
    lp = [0] * (MAX_VAL + 1)
    for p in PRIMES:
        # Step through multiples of p
        for m in range(p, MAX_VAL + 1, p):
            if lp[m] == 0:
                lp[m] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    sp = [1] * (MAX_VAL + 1)
    # sp[1] = 1
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p != 0:
            sp[m] = sp[m // p] * p
        # else sp[m] remains 1

    # Sliding window with frequency tracking for O(1) distinct count updates
    freq = {}
    distinct = 0

    # Initialize window for n=1000: indices 1001 to 1024
    # Window for n is [n+1, n+24]
    for i in range(1, WINDOW + 1):
        v = sp[1000 + i]
        count = freq.get(v, 0)
        if count == 0:
            distinct += 1
        freq[v] = count + 1

    best_n = 1000
    best_distinct = distinct

    # Scan n from 1001 to N
    # Window for n-1 was [n, n+23]
    # Window for n is [n+1, n+24]
    # Remove sp[n], Add sp[n+24]
    for n in range(1001, N + 1):
        if time.time() > deadline - 2:
            break
        
        # Remove sp[n]
        v_remove = sp[n]
        count = freq[v_remove]
        if count == 1:
            del freq[v_remove]
            distinct -= 1
        else:
            freq[v_remove] = count - 1

        # Add sp[n+24]
        v_add = sp[n + 24]
        count = freq.get(v_add, 0)
        if count == 0:
            distinct += 1
        freq[v_add] = count + 1

        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
