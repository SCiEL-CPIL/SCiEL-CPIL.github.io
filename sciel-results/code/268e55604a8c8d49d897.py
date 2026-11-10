"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    MAX_N = 1000000

    # Sieve of Eratosthenes to compute phi for all n up to MAX_N
    phi = list(range(MAX_N + 1))
    for i in range(2, MAX_N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, MAX_N + 1, i):
                phi[j] -= phi[j] // i

    # Find smallest n where phi(n) % 120 == 0
    for n in range(1, MAX_N + 1):
        if phi[n] % 120 == 0:
            return n

    # Fallback
    return 140
# EVOLVE-BLOCK-END
