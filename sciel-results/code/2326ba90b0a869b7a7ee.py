import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START

def _gen_smooth(limit):
    """Generate all 2,3,5-smooth numbers up to limit."""
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_SMOOTH_SET = set(_SMOOTH)

# Precompute divisibility compatibility masks
_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 1 << i
    for j in range(_N):
        if i == j:
            continue
        vj = _SMOOTH[j]
        if vi > vj:
            if vi % vj == 0:
                mask |= (1 << j)
        else:
            if vj % vi == 0:
                mask |= (1 << j)
    _INCOMPAT_MASK[i] = mask

# Precompute prefix sums
_PREFIX = [0] * (_N + 1)
for i in range(_N):
    _PREFIX[i + 1] = _PREFIX[i] + _SMOOTH[i]

# Global bounds
_MIN_SUM_K = [0] * (_N + 1)
_MAX_SUM_K = [0] * (_N + 1)
for k in range(1, _N + 1):
    _MIN_SUM_K[k] = _PREFIX[k]
    _MAX_SUM_K[k] = sum(_SMOOTH[-k:])

# Precompute indices sorted by value descending for faster iteration
_IDX_DESC = sorted(range(_N), key=lambda i: _SMOOTH[i], reverse=True)
# Also ascending for min-sum calculations
_IDX_ASC = sorted(range(_N), key=lambda i: _SMOOTH[i])

# Precompute for each index, the list of compatible indices (not dividing, not self)
# This helps with faster mask updates
_COMPAT_LIST = [None] * _N
for i in range(_N):
    compat = []
    for j in range(_N):
        if i == j:
            continue
        if _SMOOTH[i] > _SMOOTH[j]:
            if _SMOOTH[i] % _SMOOTH[j] != 0:
                compat.append(j)
        else:
            if _SMOOTH[j] % _SMOOTH[i] != 0:
                compat.append(j)
    _COMPAT_LIST[i] = compat

# Precompute descending index list for faster mask filtering
_IDX_DESC_LIST = sorted(range(_N), key=lambda i: _SMOOTH[i], reverse=True)

# Memoize two-term results
_TWO_CACHE = {}

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    results = [None] * 129
    total_terms = 0

    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            results[n] = _quick_fallback(n)
            total_terms += len(results[n])
            continue

        found = None
        for k in range(1, 31):
            if k > n:
                break
            if n < _MIN_SUM_K[k] or n > _MAX_SUM_K[k]:
                continue
            if k == 1:
                if n in _SMOOTH_SET:
                    found = [n]
                    break
            elif k == 2:
                found = _find_two_terms(n)
                if found is not None:
                    break
            else:
                found = _find_minimal_rep(n, k)
                if found is not None:
                    break

        if found is None:
            found = _quick_fallback(n)

        results[n] = found
        total_terms += len(found)

    return results[1:129]

def _find_two_terms(target):
    """Find two distinct smooth numbers summing to target with no divisibility."""
    if target in _TWO_CACHE:
        return _TWO_CACHE[target]
    result = None
    for i in range(_N):
        a = _SMOOTH[i]
        if a >= target:
            break
        b = target - a
        if b in _SMOOTH_SET and b != a:
            if a % b != 0 and b % a != 0:
                result = [a, b]
                break
    _TWO_CACHE[target] = result
    return result

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    """
    full_mask = (1 << _N) - 1
    path = []
    node_count = [0]
    deadline = [85.0]
    start_t = [time.time()]
    cache = {}
    desc_list = _IDX_DESC_LIST

    def dfs(remaining_k, remaining_sum, available_mask):
        if remaining_k == 0:
            return remaining_sum == 0

        if remaining_sum <= 0:
            return False

        # Check cache
        cache_key = (remaining_k, remaining_sum, available_mask)
        if cache_key in cache:
            return cache[cache_key]

        node_count[0] += 1
        if node_count[0] % 5000 == 0:
            if time.time() - start_t[0] > deadline[0]:
                cache[cache_key] = False
                return False

        # Pruning: not enough items
        if available_mask.bit_count() < remaining_k:
            cache[cache_key] = False
            return False

        # Build list of available indices using precomputed descending order
        avail = []
        for idx in desc_list:
            if (available_mask >> idx) & 1:
                avail.append(idx)

        if len(avail) < remaining_k:
            cache[cache_key] = False
            return False

        # Tight bounds: min sum = sum of smallest remaining_k available
        # max sum = sum of largest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail[-remaining_k:])
        max_sum = sum(_SMOOTH[i] for i in avail[:remaining_k])

        if remaining_sum < min_sum or remaining_sum > max_sum:
            cache[cache_key] = False
            return False

        for i in avail:
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue

            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            new_count = new_mask.bit_count()
            if new_count < remaining_k - 1:
                continue

            rem = remaining_sum - v
            if rem < _MIN_SUM_K[remaining_k - 1] or rem > _MAX_SUM_K[remaining_k - 1]:
                continue

            if remaining_k - 1 > 1:
                # Local min: sum of smallest (remaining_k-1) values in new_mask
                local_count = 0
                local_sum = 0
                for j in desc_list:
                    if (new_mask >> j) & 1:
                        local_sum += _SMOOTH[j]
                        local_count += 1
                        if local_count == remaining_k - 1:
                            break
                if local_count < remaining_k - 1:
                    continue
                if rem < local_sum:
                    continue

            path.append(i)
            if dfs(remaining_k - 1, rem, new_mask):
                cache[cache_key] = True
                return True
            path.pop()

        cache[cache_key] = False
        return False

    if dfs(k, target, full_mask):
        return [_SMOOTH[i] for i in path]
    else:
        return None

def _quick_fallback(n):
    if n in _SMOOTH_SET:
        return [n]

    found = _find_two_terms(n)
    if found is not None:
        return found

    for a in _SMOOTH:
        if a >= n:
            break
        for b in _SMOOTH:
            if b >= a:
                break
            if a + b >= n:
                break
            c = n - a - b
            if c in _SMOOTH_SET and c != a and c != b:
                if (a % b != 0 and b % a != 0 and
                    a % c != 0 and c % a != 0 and
                    b % c != 0 and c % b != 0):
                    return [a, b, c]

    # Last resort: use 1s is not valid since they must be distinct
    # Try a greedy approach with small smooth numbers
    remaining = n
    chosen = []
    for s in _SMOOTH:
        if s > remaining:
            break
        if s in chosen:
            continue
        # Check divisibility with all chosen
        valid = True
        for c in chosen:
            if c % s == 0 or s % c == 0:
                valid = False
                break
        if valid:
            chosen.append(s)
            remaining -= s
            if remaining == 0:
                return chosen

    return [1] if n == 1 else [n]

# EVOLVE-BLOCK-END
