import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    N = 123457

    # Sieve to compute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)

    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i

    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    # Scan all decompositions
    # We can iterate a from 1 to N//2 because (a, b) and (b, a) yield the same max(P(a), P(b))
    for a in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            # If we find the theoretical minimum possible, we can break early
            # The minimum possible max(P(a), P(b)) is 2 if both are powers of 2 (impossible since N is odd)
            # Or 3 if one is 1 and other is 3*2^k? No, 123456 is not 3*2^k.
            # Actually, if we find max(P(a), P(b)) = 2, that's impossible for odd N > 1.
            # If we find 3, it means both a and b are 3-smooth? No, max is 3 means all prime factors <= 3.
            # So a and b must be of form 2^i * 3^j.
            # This is a very strong condition. Let's just run the full loop as N is small enough.
            
    return [best_a, best_b]
