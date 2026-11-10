import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 10000000

    # Linear sieve for sigma (multiplicative function)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sp = np.zeros(N + 1, dtype=np.int64)  # smallest prime factor
    pe = np.zeros(N + 1, dtype=np.int64)  # highest power of smallest prime dividing n
    sigma[1] = 1
    primes = []

    for i in range(2, N + 1):
        if time.time() - start_time > 25.0:
            break
        if sp[i] == 0:
            sp[i] = i
            pe[i] = i
            sigma[i] = 1 + i
            primes.append(i)
        for p in primes:
            if p > sp[i] or i * p > N:
                break
            sp[i * p] = p
            if i % p == 0:
                pe[i * p] = pe[i] * p
                m = i // pe[i]
                sigma[i * p] = sigma[i] + sigma[m] * pe[i] * p
                break
            else:
                pe[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)

    # Search for pairs (a, b) with sigma[a] == sigma[b] minimizing |a/b - 3/2|
    best_error = float('inf')
    best_a = 1
    best_b = 1

    # Initial bound from known pair: sigma(11)=12=sigma(6), error=|11/6-3/2|=1/6
    if N >= 11 and sigma[11] == sigma[6]:
        best_error = 1.0 / 6.0
        best_a, best_b = 11, 6

    # Search b from N down to 1
    for b in range(N, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        # Pruning: min possible error for this b is 1/(2b)
        if best_error < float('inf') and 1.0 / (2.0 * b) >= best_error:
            break

        # Check candidates near a = 1.5*b
        center = (3 * b) // 2
        for da in range(-5, 6):
            a = center + da
            if a < 1 or a > N:
                continue
            if 2 * a == 3 * b:
                continue  # error would be 0, invalid
            if sigma[a] == sigma[b]:
                err = abs(2 * a - 3 * b) / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]
