"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import math

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    N = 1000001  # need pp up to n+1 where n<=1000000
    LO = 10000
    HI = 1000000

    # Smallest prime factor sieve
    spf = list(range(N + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:  # prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # powerful part for each n
    pp = [1] * (N + 1)
    for n in range(2, N + 1):
        x = n
        prod = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                prod *= p ** e
        pp[n] = prod

    # scan consecutive pairs n in [LO, HI]
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    for n in range(LO, HI + 1):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END
