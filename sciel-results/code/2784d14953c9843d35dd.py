"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 88.0  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1

    # Sieve for largest prime factor:
    # lpf[m] = largest prime factor of m.
    # Iterate p from 2 to limit; if lpf[p]==0, p is prime.
    # For each prime p, set lpf[m]=p for all multiples m of p.
    # Since p increases, the last assignment is the largest prime factor.
    lpf = [0] * (limit + 1)
    for p in range(2, limit + 1):
        if lpf[p] == 0:
            for m in range(p, limit + 1, p):
                lpf[m] = p

    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = lpf[N_MIN]
    if lpf[N_MIN + 1] > best_val:
        best_val = lpf[N_MIN + 1]

    check_interval = 50000
    for n in range(N_MIN + 1, N_MAX + 1):
        val = lpf[n]
        if lpf[n + 1] > val:
            val = lpf[n + 1]
        if val < best_val:
            best_val = val
            best_n = n

        if (n - N_MIN) % check_interval == 0:
            if time.time() > deadline:
                break

    return best_n
# EVOLVE-BLOCK-END
