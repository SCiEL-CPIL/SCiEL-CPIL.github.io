# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for p in range(2, N + 1):
        if lpf[p] == 0:  # p is prime
            for k in range(p, N + 1, p):
                lpf[k] = p
    
    # Find a that minimizes max(P(a), P(N-a))
    best_max = N + 1
    best_a = 1
    for a in range(1, N):
        b = N - a
        m = lpf[a]
        if lpf[b] > m:
            m = lpf[b]
        if m < best_max:
            best_max = m
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
