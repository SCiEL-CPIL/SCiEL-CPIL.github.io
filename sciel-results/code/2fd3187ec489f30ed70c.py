import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    M = N_MAX + 1  # we need lpf for values up to 1000001

    # Sieve of Eratosthenes to find primes up to M
    is_prime = bytearray(b'\x01') * (M + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(M) + 1):
        if is_prime[i]:
            is_prime[i*i : M+1 : i] = bytearray(len(range(i*i, M+1, i)))

    # Collect primes
    primes = [i for i in range(2, M + 1) if is_prime[i]]

    # Compute largest prime factor for each number up to M
    # lpf[x] = largest prime factor of x
    lpf = [0] * (M + 1)
    for p in primes:
        # For each multiple of p, set lpf to p (since p is increasing, last one wins)
        for multiple in range(p, M + 1, p):
            lpf[multiple] = p

    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > DEADLINE:
            break

    return best_n
