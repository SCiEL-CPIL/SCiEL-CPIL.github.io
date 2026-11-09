"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import defaultdict
import numpy as np
import bisect

_MAX = 10_000_000

def _build_sigma_sieve(n):
    """Compute sigma(k) for all 1 <= k <= n via divisor-sum sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def construct():
    t_start = time.time()
    deadline = t_start + 70.0

    sigma = _build_sigma_sieve(_MAX)
    sigma_list = sigma.tolist()

    groups = defaultdict(list)
    for i in range(1, _MAX + 1):
        groups[sigma_list[i]].append(i)

    best_error = Fraction(1, 3)
    best_pair = (11, 6)

    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        if time.time() > deadline:
            break
        lst.sort()
        for b in lst:
            target_a = (3 * b) // 2
            pos = bisect.bisect_left(lst, target_a)
            for a in (lst[pos] if pos < len(lst) else -1,
                      lst[pos-1] if pos > 0 else -1,
                      lst[pos+1] if pos+1 < len(lst) else -1,
                      lst[pos-2] if pos-2 >= 0 else -1):
                if a < 1 or a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                err = Fraction(num, 2 * b)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)

    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
