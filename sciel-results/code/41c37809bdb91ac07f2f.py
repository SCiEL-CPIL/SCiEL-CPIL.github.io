"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_divs = _get_divisors(5040)
_n_divs = len(_divs)
_target = 4979
_div_set = set(_divs)
_divs_desc = sorted(_divs, reverse=True)

def _max_sum_k(k):
    """Sum of the k largest divisors."""
    if k <= 0:
        return 0
    if k >= _n_divs:
        return sum(_divs)
    return sum(_divs_desc[:k])

def _min_sum_k(k):
    """Sum of the k smallest divisors."""
    if k <= 0:
        return 0
    if k >= _n_divs:
        return sum(_divs)
    return sum(_divs[:k])

def _find_subset_k(k, deadline):
    n = _n_divs
    target = _target

    # Pruning: if even the k largest divisors can't reach target, impossible
    if _max_sum_k(k) < target:
        return None
    # If the k smallest divisors already exceed target, impossible (since all divisors positive)
    if _min_sum_k(k) > target:
        return None

    if k == 1:
        if target in _div_set:
            return [target]
        return None

    if k == 2:
        for d in _divs:
            if d >= target:
                break
            complement = target - d
            if complement in _div_set and complement != d:
                return [d, complement]
        return None

    if k == 3:
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                s2 = _divs[i] + _divs[j]
                if s2 >= target:
                    break
                complement = target - s2
                if complement in _div_set and complement > _divs[j]:
                    return [_divs[i], _divs[j], complement]
        return None

    if k == 4:
        pair_sums = defaultdict(list)
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                s = _divs[i] + _divs[j]
                if s < target:
                    pair_sums[s].append((i, j))
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                s = _divs[i] + _divs[j]
                if s >= target:
                    break
                complement = target - s
                if complement in pair_sums:
                    for (p, q) in pair_sums[complement]:
                        if len({i, j, p, q}) == 4:
                            return [_divs[i], _divs[j], _divs[p], _divs[q]]
        return None

    if k == 5:
        triple_sums = defaultdict(list)
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                for m in range(j + 1, n):
                    s = _divs[i] + _divs[j] + _divs[m]
                    if s < target:
                        triple_sums[s].append((i, j, m))
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                s2 = _divs[i] + _divs[j]
                if s2 >= target:
                    break
                complement = target - s2
                if complement in triple_sums:
                    for (p, q, r) in triple_sums[complement]:
                        if len({i, j, p, q, r}) == 5:
                            return [_divs[i], _divs[j], _divs[p], _divs[q], _divs[r]]
        return None

    if k == 6:
        triple_sums = defaultdict(list)
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                for m in range(j + 1, n):
                    s = _divs[i] + _divs[j] + _divs[m]
                    if s < target:
                        triple_sums[s].append((i, j, m))
        for i in range(n):
            if time.time() > deadline:
                return None
            for j in range(i + 1, n):
                for m in range(j + 1, n):
                    s = _divs[i] + _divs[j] + _divs[m]
                    if s >= target:
                        break
                    complement = target - s
                    if complement in triple_sums:
                        for (p, q, r) in triple_sums[complement]:
                            if len({i, j, m, p, q, r}) == 6:
                                return [_divs[i], _divs[j], _divs[m], _divs[p], _divs[q], _divs[r]]
        return None

    # k >= 7: greedy
    remainder = target
    chosen = []
    for d in _divs_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            return chosen
    return None

def construct():
    deadline = time.time() + 55.0

    for k in range(1, 257):
        if time.time() > deadline:
            break
        result = _find_subset_k(k, deadline)
        if result is not None:
            return result

    # Fallback: greedy
    remainder = _target
    chosen = []
    for d in _divs_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
# EVOLVE-BLOCK-END
