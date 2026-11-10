import time

def construct():
    start = time.time()
    DEADLINE = 60.0

    N = 1000000
    NMAX = N + 1

    # --- Sieve primes up to NMAX using bytearray ---
    is_prime = bytearray(b'\x01') * (NMAX + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    limit = int(NMAX ** 0.5) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i * i :: i] = b'\x00' * (((NMAX - i * i) // i) + 1)

    # --- Compute Largest Prime Factor for every integer up to NMAX ---
    # We use a list of lists of integers for speed, though a bytearray/array would also work.
    # Actually, since NMAX is ~1e6, a list of ints is fine and often faster for small updates.
    # Let's use a standard list of zeros.
    lpf = [0] * (NMAX + 1)
    
    # Iterate through potential primes in increasing order
    # If is_prime[p] is true, p is a prime. Stamp p onto all multiples.
    for p in range(2, NMAX + 1):
        if is_prime[p]:
            for m in range(p, NMAX + 1, p):
                lpf[m] = p

    # --- Scan the window for the minimum of max(lpf[n], lpf[n+1]) ---
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])

    for n in range(10001, N + 1):
        val = lpf[n]
        v2 = lpf[n + 1]
        if v2 > val:
            val = v2
        if val < best_val:
            best_val = val
            best_n = n
        
        # Periodic deadline guard
        if (n & 0xFFFF) == 0 and time.time() > DEADLINE - 2.0:
            break

    return best_n
