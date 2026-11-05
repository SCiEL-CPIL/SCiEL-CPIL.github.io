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

# Precompute divisibility incompatibility masks
_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 1 << i  # Incompatible with itself (distinctness)
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

def _check_div_free(vals):
    """Check if all pairs in vals are divisibility-free."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if vals[i] % vals[j] == 0 or vals[j] % vals[i] == 0:
                return False
    return True

def _find_rep_bruteforce(target, max_k=6):
    """Find a valid representation using brute force for small k."""
    if target in _SMOOTH_SET:
        return [target]
    
    # Try 2 terms
    for a in _SMOOTH:
        if a >= target:
            break
        b = target - a
        if b in _SMOOTH_SET and a != b and a % b != 0 and b % a != 0:
            return [a, b]
    
    # Try 3 terms
    for i, a in enumerate(_SMOOTH):
        if a >= target:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= target:
                break
            c = target - a - b
            if c in _SMOOTH_SET and c > b:
                if _check_div_free([a, b, c]):
                    return [a, b, c]
    
    # Try 4 terms
    for i, a in enumerate(_SMOOTH):
        if a >= target:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= target:
                break
            for k in range(j + 1, _N):
                c = _SMOOTH[k]
                if a + b + c >= target:
                    break
                d = target - a - b - c
                if d in _SMOOTH_SET and d > c:
                    if _check_div_free([a, b, c, d]):
                        return [a, b, c, d]
    
    # Try 5 terms
    for i, a in enumerate(_SMOOTH):
        if a >= target:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= target:
                break
            for k in range(j + 1, _N):
                c = _SMOOTH[k]
                if a + b + c >= target:
                    break
                for l in range(k + 1, _N):
                    d = _SMOOTH[l]
                    if a + b + c + d >= target:
                        break
                    e = target - a - b - c - d
                    if e in _SMOOTH_SET and e > d:
                        if _check_div_free([a, b, c, d, e]):
                            return [a, b, c, d, e]
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    results = [None] * 129
    total_terms = 0

    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            results[n] = _find_rep_bruteforce(n, 6)
            if results[n] is None:
                results[n] = [n]  # Defensive
            total_terms += len(results[n])
            continue

        found = None
        # Try k from 1 to 30
        for k in range(1, 31):
            if k > n:
                break
            # Feasibility: minimum sum of k distinct positive integers is k*(k+1)//2
            if n < k * (k + 1) // 2:
                break
            
            # For small k, use brute force (faster for k<=5)
            if k <= 5:
                found = _find_rep_bruteforce(n, k)
                if found is not None and len(found) == k:
                    break
                elif found is not None and len(found) < k:
                    # Found a smaller representation, use it
                    break
                else:
                    continue
            else:
                found = _find_minimal_rep(n, k)
                if found is not None:
                    break

        if found is None:
            found = _find_rep_bruteforce(n, 6)
            if found is None:
                found = [n]  # Defensive

        results[n] = found
        total_terms += len(found)

    return results[1:129]

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    """
    if k == 1:
        if target in _SMOOTH_SET:
            return [target]
        return None

    # DFS with pruning using bitmask
    def dfs(remaining_k, remaining_sum, available_mask, current_path):
        if remaining_k == 0:
            return remaining_sum == 0
        
        if remaining_sum <= 0:
            return False
            
        # Get available indices sorted by value (ascending)
        avail_indices = []
        m = available_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            m ^= lsb
            
        if len(avail_indices) < remaining_k:
            return False
            
        # Pruning: min possible sum = sum of smallest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
        if remaining_sum < min_sum:
            return False
            
        # Pruning: max possible sum = sum of largest remaining_k available
        max_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
        if remaining_sum > max_sum:
            return False
        
        for i in avail_indices:
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue
            new_remaining_sum = remaining_sum - v
            if new_remaining_sum < 0:
                continue
            # Minimum sum of (remaining_k-1) distinct positive integers
            if new_remaining_sum < (remaining_k - 1) * remaining_k // 2:
                continue
                
            current_path.append(i)
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            
            if dfs(remaining_k - 1, new_remaining_sum, new_mask, current_path):
                return True
                
            current_path.pop()
            
        return False

    full_mask = (1 << _N) - 1
    path = []
    
    if dfs(k, target, full_mask, path):
        return [_SMOOTH[i] for i in path]
    else:
        return None

# EVOLVE-BLOCK-END
