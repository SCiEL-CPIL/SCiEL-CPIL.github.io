import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Precompute the small-prime factor (the part of x composed of primes < 24)
    # for all x up to 1000000 + 24
    max_val = 1000000 + 24
    sf = [1] * (max_val + 1)

    # Sieve-style: for each prime p < 24, multiply sf[x] by p for every x divisible by p
    for p in small_primes:
        for x in range(p, max_val + 1, p):
            sf[x] *= p

    # Now scan for the best n in [1000, 1000000]
    best_score = 25
    best_n = 1000

    for n in range(1000, 1000001):
        if time.time() > deadline:
            break

        # Collect the 24 small-prime factors
        factors = [sf[n + i] for i in range(1, 25)]
        distinct_count = len(set(factors))

        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break  # can't do better than 1

    return best_n
