"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Precompute smallest prime factor (SPF) up to 10^6 using a modified sieve
    LIMIT = 1000000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def phi(n):
        if n == 1:
            return 1
        result = n
        m = n
        p = spf[m]
        while m > 1:
            result -= result // p
            while m % p == 0:
                m //= p
            if m > 1:
                p = spf[m]
        return result

    # Search for the smallest n in [1, 1000000] with phi(n) % 120 == 0
    for n in range(1, LIMIT + 1):
        if phi(n) % 120 == 0:
            return n
    return 143  # Fallback (should never be reached)

# EVOLVE-BLOCK-END
