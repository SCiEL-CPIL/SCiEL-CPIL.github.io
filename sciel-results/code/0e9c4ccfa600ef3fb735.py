import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 10000000

    # Compute smallest prime factor for all n up to N
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i

    # Compute sigma(n) for all n using SPF
    sigma = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        if time.time() > deadline - 20.0:
            break
        p = int(spf[n])
        m = n
        k = 0
        while m % p == 0:
            m //= p
            k += 1
        # sigma(n) = sigma(m) * (1 + p + p^2 + ... + p^k)
        geom = 1
        p_power = 1
        for _ in range(k):
            p_power *= p
            geom += p_power
        sigma[n] = sigma[m] * geom

    if time.time() > deadline - 15.0:
        return [11, 6]

    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)

    # Find best pair using binary search within each group
    best_error = Fraction(1, 3)
    best_pair = (11, 6)

    # Sort groups by size descending to check larger groups first (more likely to have good pairs)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

    for s, lst in sorted_groups:
        if time.time() > deadline - 5.0:
            break
        if len(lst) < 2:
            continue
        lst.sort()
        for b in lst:
            target = 3 * b / 2.0
            pos = bisect.bisect_left(lst, target)
            for idx in [pos - 1, pos]:
                if 0 <= idx < len(lst):
                    a = lst[idx]
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    err = Fraction(num, den)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
