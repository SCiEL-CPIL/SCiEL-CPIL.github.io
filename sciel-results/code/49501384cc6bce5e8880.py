import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    N_MAX = 5_000_000

    # 1. Prime sieve using NumPy
    is_prime = np.ones(N_MAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]

    # 2. SPF array
    spf = np.zeros(N_MAX + 1, dtype=np.int64)
    for p in primes:
        if p * p > N_MAX:
            break
        multiples = np.arange(p * p, N_MAX + 1, p)
        mask = spf[multiples] == 0
        spf[multiples[mask]] = p
    # For primes, spf[p] = p
    spf[is_prime] = np.where(is_prime)[0]
    spf[1] = 1

    # 3. Sigma computation using SPF recurrence
    p_power = np.ones(N_MAX + 1, dtype=np.int64)
    rest = np.ones(N_MAX + 1, dtype=np.int64)
    sigma = np.zeros(N_MAX + 1, dtype=np.int64)
    sigma[1] = 1

    for i in range(2, N_MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            p_power[i] = p * p_power[m]
            rest[i] = rest[m]
        else:
            p_power[i] = p
            rest[i] = m

        pp = p_power[i]
        if p == 2:
            geom = 2 * pp - 1
        else:
            geom = (p * pp - 1) // (p - 1)

        sigma[i] = sigma[rest[i]] * geom

        if time.time() > deadline - 25.0:
            break

    # 4. Count collisions
    counts = {}
    for k in range(1, N_MAX + 1):
        M = int(k) * int(sigma[k])
        if M > 10**18:
            continue
        if M in counts:
            counts[M] += 1
        else:
            counts[M] = 1

        if time.time() > deadline - 5.0:
            break

    if not counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_M = max(counts, key=counts.get)

    # 5. Collect k's for best_M
    k_list = []
    limit = math.isqrt(best_M)
    upper = min(N_MAX, limit)

    for k in range(1, upper + 1):
        if int(k) * int(sigma[k]) == best_M:
            k_list.append(k)

    k_list = sorted(set(k_list))

    if not k_list:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_M), 'k': [int(k) for k in k_list]}
