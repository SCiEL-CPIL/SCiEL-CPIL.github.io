import time

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
            found = _find_minimal_rep(n, k)
            if found is not None:
                break

        if found is None:
            found = _quick_fallback(n)

        results[n] = found
        total_terms += len(found)

    return results[1:129]

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

    def dfs(remaining_k, remaining_sum, available_mask):
        if remaining_k == 0:
            return remaining_sum == 0

        if remaining_sum <= 0:
            return False

        node_count[0] += 1
        if node_count[0] % 10000 == 0:
            if time.time() - start_t[0] > deadline[0]:
                return False

        # Pruning: not enough items
        if available_mask.bit_count() < remaining_k:
            return False

        # Build list of available indices from mask, sorted descending by value
        avail = []
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            avail.append(idx)
            m ^= lsb

        if len(avail) < remaining_k:
            return False

        # Sort descending by value
        avail.sort(key=lambda i: _SMOOTH[i], reverse=True)

        # Tight bounds: min sum = sum of smallest remaining_k available
        # max sum = sum of largest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail[-remaining_k:])
        max_sum = sum(_SMOOTH[i] for i in avail[:remaining_k])

        if remaining_sum < min_sum or remaining_sum > max_sum:
            return False

        # Additional pruning: the largest single element must be <= remaining_sum
        # and we need remaining_sum - largest >= min possible sum of (remaining_k-1) smallest
        for pos, i in enumerate(avail):
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue

            # After picking i, we need remaining_k-1 more elements summing to remaining_sum - v
            # from avail[pos+1:] (those that are compatible and haven't been tried yet in this branch)
            # But actually all of avail except i are candidates, minus those incompatible with i
            # The new available set is available_mask & ~_INCOMPAT_MASK[i]
            # For pruning: the min sum of (remaining_k-1) elements from the new available set
            # must be <= remaining_sum - v

            # Quick check: if remaining_k > 1, the smallest possible next element
            # is at least the smallest in the new available set
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            new_count = new_mask.bit_count()
            if new_count < remaining_k - 1:
                continue

            # Compute min and max possible sums for remaining_k-1 from new_mask
            # This is expensive; do a quick bound using global min/max
            # Min possible sum of (remaining_k-1) distinct smooth numbers >= _MIN_SUM_K[remaining_k-1]
            # Max possible sum <= _MAX_SUM_K[remaining_k-1]
            rem = remaining_sum - v
            if rem < _MIN_SUM_K[remaining_k - 1] or rem > _MAX_SUM_K[remaining_k - 1]:
                continue

            # Even tighter: extract the smallest (remaining_k-1) from new_mask
            if remaining_k - 1 > 1:
                # Get the smallest (remaining_k-1) values from new_mask
                nm = new_mask
                small_vals = []
                while nm and len(small_vals) < remaining_k - 1:
                    lsb = nm & -nm
                    idx = lsb.bit_length() - 1
                    small_vals.append(_SMOOTH[idx])
                    nm ^= lsb
                if len(small_vals) < remaining_k - 1:
                    continue
                local_min = sum(small_vals)
                if rem < local_min:
                    continue

            path.append(i)
            if dfs(remaining_k - 1, remaining_sum - v, new_mask):
                return True
            path.pop()

        return False

    if dfs(k, target, full_mask):
        return [_SMOOTH[i] for i in path]
    else:
        return None

def _quick_fallback(n):
    if n in _SMOOTH_SET:
        return [n]

    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b in _SMOOTH_SET and a != b:
            if a % b != 0 and b % a != 0:
                return [a, b]

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

    return [1] if n == 1 else [n]

# EVOLVE-BLOCK-END
