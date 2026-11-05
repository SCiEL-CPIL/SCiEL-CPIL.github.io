import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before the 90s kill

TARGET = factorial(12)  # 479001600

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

# Precompute all divisors of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
def _all_divisors():
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divs = [1]
    for i, e in enumerate(exps):
        p = primes[i]
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    return divs

_ALL_DIVS = _all_divisors()

# For each divisor, precompute its "signature" as a tuple of exponents
# This lets us check distinctness and product constraints quickly
def _exponent_tuple(n):
    primes = [2, 3, 5, 7, 11]
    t = []
    for p in primes:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        t.append(e)
    if n != 1:
        return None
    return tuple(t)

# Map each divisor to its exponent tuple
_DIV_EXP = {}
for d in _ALL_DIVS:
    t = _exponent_tuple(d)
    if t is not None:
        _DIV_EXP[d] = t

# Filter divisors > 12
_VALID_DIVS = [d for d in _ALL_DIVS if d > 12]

def _divisors_in_range(rem, lo, hi):
    """Return sorted divisors of rem in [lo, hi]."""
    result = []
    for d in _VALID_DIVS:
        if d < lo:
            continue
        if d > hi:
            break
        if rem % d == 0:
            result.append(d)
    return result

def _dfs(rem, min_val, chosen, path, best_max, best_path):
    """
    DFS to find factors.
    rem: product remaining
    min_val: smallest factor allowed (for distinctness/ordering)
    chosen: set of chosen factors
    path: list of factors in current path
    best_max: [current best max factor]
    best_path: [current best path]
    """
    if _time_left() < 1.0:
        return False

    if rem == 1:
        if len(path) >= 1:
            current_max = max(path)
            if best_max[0] is None or current_max < best_max[0]:
                best_max[0] = current_max
                best_path[0] = list(path)
        return True

    if len(path) >= 12:
        return False

    # Bound: if we already have a best, don't try factors >= best_max
    hi = best_max[0] - 1 if best_max[0] is not None else TARGET
    if hi < min_val:
        return False

    # Pruning: if rem < min_val, no factor can be small enough
    if rem < min_val:
        return False

    # Get divisors in range [min_val, hi]
    divs = _divisors_in_range(rem, min_val, hi)

    if not divs:
        return False

    # Try each candidate
    for d in divs:
        if d in chosen:
            continue

        new_rem = rem // d

        # If new_rem == 1, we're done (handled at top of next call)
        # If new_rem < d+1, we can't form remaining product with factors >= d+1
        if new_rem < d + 1 and new_rem != 1:
            continue

        # If new_rem > 1, we need at least one more factor
        # The next factor must be >= d+1 and <= hi
        if new_rem > 1:
            if new_rem > hi:
                # Need to split new_rem into factors each in [d+1, hi]
                # At minimum, need factors of size >= d+1
                # If new_rem < (d+1)^2, we can't split it into 2+ factors each >= d+1
                # unless new_rem itself is a valid single factor
                if new_rem < (d + 1) * (d + 1):
                    # Can only use new_rem as a single remaining factor
                    if new_rem <= hi and new_rem > 12 and new_rem not in chosen and new_rem >= d + 1:
                        chosen.add(d)
                        path.append(d)
                        chosen.add(new_rem)
                        path.append(new_rem)
                        current_max = max(path)
                        if best_max[0] is None or current_max < best_max[0]:
                            best_max[0] = current_max
                            best_path[0] = list(path)
                        path.pop()
                        chosen.remove(new_rem)
                        path.pop()
                        chosen.remove(d)
                    continue
                # Otherwise, need at least 2 more factors, each >= d+1
                min_prod_2 = (d + 1) * (d + 2)
                if new_rem < min_prod_2:
                    continue

        chosen.add(d)
        path.append(d)

        if _dfs(new_rem, d + 1, chosen, path, best_max, best_path):
            return True

        path.pop()
        chosen.remove(d)

    return False

def construct():
    best_max = [None]
    best_path = [None]

    chosen = set()
    path = []
    _dfs(TARGET, 13, chosen, path, best_max, best_path)

    if best_path[0] is not None:
        return best_path[0]

    return [TARGET]
