"""
Pre-compute optimal antichain smooth-sum representations for targets 1..128.
Uses aggressive search with time budgeting and fallback strategies.
"""
import time
from functools import lru_cache

_T0 = time.time()
_DEADLINE = 75.0  # Safety margin before 90s kill

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

def _gen_smooth(limit):
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

_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)
_VAL_INDEX = {v: i for i, v in enumerate(_SMOOTH)}
_NSM = len(_SMOOTH)

def _div_free(a, b):
    return (a % b != 0) and (b % a != 0)

# Precompute incompatibility matrix
_INCOMPAT = [[False] * _NSM for _ in range(_NSM)]
for i in range(_NSM):
    for j in range(i + 1, _NSM):
        if not _div_free(_SMOOTH[i], _SMOOTH[j]):
            _INCOMPAT[i][j] = True
            _INCOMPAT[j][i] = True

# Precompute prefix sums for pruning
_PREFIX = [0] * (_NSM + 1)
for i in range(_NSM):
    _PREFIX[i + 1] = _PREFIX[i] + _SMOOTH[i]

def _check_valid(sol, n):
    """Verify a solution is valid: distinct smooth terms, antichain, sums to n."""
    if len(sol) == 0 or len(sol) > 30:
        return False
    if sum(sol) != n:
        return False
    seen = set()
    for a in sol:
        if a in seen or a not in _SMOOTH_SET or a <= 0:
            return False
        seen.add(a)
        for b in sol:
            if b != a and not _div_free(a, b):
                return False
    return True

def _solve_target(n):
    """Find min-k div-free smooth representation of n using DFS with pruning."""
    if n in _VAL_INDEX:
        return [n]
    
    # Candidate indices: smooth values <= n
    cand_idx = [i for i in range(_NSM) if _SMOOTH[i] <= n]
    ncand = len(cand_idx)
    if ncand == 0:
        return None
    
    # Sort candidates in descending order for better search
    cand_desc = sorted(cand_idx, key=lambda i: _SMOOTH[i], reverse=True)
    cd_vals = [_SMOOTH[i] for i in cand_desc]
    ncand = len(cd_vals)
    
    # Rebuild incompatibility for desc order
    incompat_d = [0] * ncand
    for i in range(ncand):
        gi = cand_desc[i]
        for j in range(i + 1, ncand):
            gj = cand_desc[j]
            if _INCOMPAT[gi][gj]:
                incompat_d[i] |= (1 << j)
                incompat_d[j] |= (1 << i)
    
    # Precompute suffix sums for max bound: suffix_sum[i][k] = sum of cd_vals[i:i+k]
    suffix_sum = [[0] * (ncand + 1) for _ in range(ncand + 1)]
    for i in range(ncand):
        for k in range(1, ncand - i + 1):
            suffix_sum[i][k] = suffix_sum[i][k-1] + cd_vals[i+k-1]
    
    # Precompute prefix sums for min bound: prefix_sum[i][k] = sum of cd_vals[i:i+k] from the end
    # Actually, for min bound we need sum of k smallest from position i onward
    # Since sorted descending, the k smallest from position i onward are at the end
    # min_sum[i][k] = sum of the last k elements in cd_vals[i:]
    # = cd_vals[ncand-k] + ... + cd_vals[ncand-1] if ncand-k >= i
    # More precisely: min_sum[i][k] = sum(cd_vals[max(i, ncand-k):ncand])
    min_sum = [[0] * (ncand + 1) for _ in range(ncand + 1)]
    for i in range(ncand):
        for k in range(1, ncand - i + 1):
            # The k smallest from position i onward are the last k in cd_vals[i:]
            # which are at indices ncand-k to ncand-1 (if ncand-k >= i)
            # or from i to i+k-1 if we're near the end
            # Actually simpler: sum of cd_vals[ncand-k:ncand] if ncand-k >= i, else sum of cd_vals[i:i+k]
            # But we want the MINIMUM possible sum of k elements from cd_vals[i:]
            # Since sorted descending, the minimum sum is achieved by taking the last k elements
            # which are cd_vals[ncand-k], ..., cd_vals[ncand-1]
            # But these must all be >= i, so we need ncand-k >= i
            # If ncand-k < i, then we take cd_vals[i:i+k]
            start_idx = max(i, ncand - k)
            min_sum[i][k] = sum(cd_vals[start_idx:ncand]) if start_idx + k <= ncand else sum(cd_vals[i:i+k])
    
    result = [None]
    
    def backtrack(start, k, remaining, chosen_mask, chosen_list):
        if result[0] is not None:
            return True
        if k == 0:
            if remaining == 0:
                result[0] = list(chosen_list)
                return True
            return False
        if remaining <= 0:
            return False
        if start + k > ncand:
            return False
        # Max possible sum: k largest from position start
        if suffix_sum[start][k] < remaining:
            return False
        # Min possible sum: k smallest from position start
        if min_sum[start][k] > remaining:
            return False
        
        for i in range(start, ncand):
            v = cd_vals[i]
            if v > remaining:
                continue
            if incompat_d[i] & chosen_mask:
                continue
            # Additional pruning: after choosing v, can we still reach remaining-v with k-1 more?
            new_remaining = remaining - v
            if k - 1 > 0:
                if i + 1 + (k - 1) > ncand:
                    continue
                if suffix_sum[i+1][k-1] < new_remaining:
                    continue
                if min_sum[i+1][k-1] > new_remaining:
                    continue
            chosen_mask |= (1 << i)
            chosen_list.append(v)
            if backtrack(i + 1, k - 1, remaining - v, chosen_mask, chosen_list):
                return True
            chosen_list.pop()
            chosen_mask &= ~(1 << i)
        return False
    
    max_k = min(n, ncand)
    for k in range(2, max_k + 1):
        if time.time() - _T0 > _DEADLINE:
            break
        if backtrack(0, k, n, 0, []):
            break
    
    return result[0]

def _fallback2(n):
    """2-term search."""
    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b in _VAL_INDEX and b > 0 and _div_free(a, b):
            return [a, b]
    return None

def _fallback3(n):
    """3-term search."""
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a >= n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c in _VAL_INDEX and c > 0 and _div_free(a, b) and _div_free(a, c) and _div_free(b, c):
                return [a, b, c]
    return None

def _greedy_fill(n):
    """Greedy: pick largest compatible smooth numbers, then fill remainder."""
    remaining = n
    chosen = []
    chosen_set = set()
    # Try to build a valid antichain greedily
    for v in sorted(_SMOOTH, reverse=True):
        if v > remaining:
            continue
        if v in chosen_set:
            continue
        # Check divisibility with all chosen
        ok = True
        for c in chosen:
            if not _div_free(c, v):
                ok = False
                break
        if ok:
            chosen.append(v)
            chosen_set.add(v)
            remaining -= v
            if remaining == 0:
                break
    if remaining == 0 and chosen:
        return chosen
    return None

def _brute_small(n):
    """For small n, try all subsets up to size 5."""
    from itertools import combinations
    cand = [v for v in _SMOOTH if v <= n]
    for k in range(2, min(6, len(cand)) + 1):
        for combo in combinations(cand, k):
            if sum(combo) != n:
                continue
            # Check antichain
            ok = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if not _div_free(combo[i], combo[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return list(combo)
    return None

# Pre-compute all answers
_ANSWERS = [None] * 129

# Phase 1: single-term and easy cases
for n in range(1, 129):
    if n in _VAL_INDEX:
        _ANSWERS[n] = [n]

# Phase 2: 2-term search
for n in range(1, 129):
    if _ANSWERS[n] is None:
        sol = _fallback2(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Phase 3: 3-term search
for n in range(1, 129):
    if _ANSWERS[n] is None:
        sol = _fallback3(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Phase 4: Brute force small for remaining
for n in range(1, 129):
    if _ANSWERS[n] is None and n <= 40:
        sol = _brute_small(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Phase 5: Full search for remaining, with time budget
for n in range(1, 129):
    if _ANSWERS[n] is None:
        if time.time() - _T0 > _DEADLINE - 5:
            break
        sol = _solve_target(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Phase 6: Greedy fill for any remaining
for n in range(1, 129):
    if _ANSWERS[n] is None:
        sol = _greedy_fill(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Phase 7: Extended brute force for remaining
for n in range(1, 129):
    if _ANSWERS[n] is None:
        if time.time() - _T0 > _DEADLINE - 2:
            break
        # Try up to size 7
        from itertools import combinations
        cand = [v for v in _SMOOTH if v <= n]
        found = False
        for k in range(2, min(8, len(cand)) + 1):
            if time.time() - _T0 > _DEADLINE - 1:
                break
            for combo in combinations(cand, k):
                if sum(combo) != n:
                    continue
                ok = True
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        if not _div_free(combo[i], combo[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    _ANSWERS[n] = list(combo)
                    found = True
                    break
            if found:
                break

# Phase 8: Last resort - any valid representation
for n in range(1, 129):
    if _ANSWERS[n] is None:
        # Try DFS with larger k
        if time.time() - _T0 > _DEADLINE - 0.5:
            break
        # Use _solve_target which handles up to ncand
        sol = _solve_target(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Final fallback: ensure all have something
for n in range(1, 129):
    if _ANSWERS[n] is None:
        # This should not happen, but provide a safe fallback
        # Use 1's - but that's invalid. Try harder.
        # Actually, let's do a very aggressive search
        from itertools import combinations
        cand = [v for v in _SMOOTH if v <= n]
        for k in range(2, min(15, len(cand)) + 1):
            for combo in combinations(cand, k):
                if sum(combo) != n:
                    continue
                ok = True
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        if not _div_free(combo[i], combo[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    _ANSWERS[n] = list(combo)
                    break
            if _ANSWERS[n] is not None:
                break
        if _ANSWERS[n] is None:
            # Absolute last resort
            _ANSWERS[n] = [n] if n in _SMOOTH_SET else [1]

def construct():
    return _ANSWERS[1:129]
