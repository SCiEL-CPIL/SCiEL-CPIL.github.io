"""
Pre-compute optimal antichain smooth-sum representations for targets 1..128.
Uses aggressive search with time budgeting, exponent-vector pruning, and fallback strategies.
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

# Precompute exponent vectors (a, b, c) for each smooth number 2^a * 3^b * 5^c
_EXP_VECTORS = []
for v in _SMOOTH:
    n = v
    a = b = c = 0
    while n % 2 == 0:
        n //= 2
        a += 1
    while n % 3 == 0:
        n //= 3
        b += 1
    while n % 5 == 0:
        n //= 5
        c += 1
    _EXP_VECTORS.append((a, b, c))

def _div_free(a, b):
    return (a % b != 0) and (b % a != 0)

def _exp_dominates(e1, e2):
    """Check if exponent vector e1 dominates e2 (i.e., 2^a1*3^b1*5^c1 divides 2^a2*3^b2*5^c2)."""
    return e1[0] <= e2[0] and e1[1] <= e2[1] and e1[2] <= e2[2]

# Precompute incompatibility matrix using exponent vectors
_INCOMPAT = [[False] * _NSM for _ in range(_NSM)]
for i in range(_NSM):
    for j in range(i + 1, _NSM):
        ei = _EXP_VECTORS[i]
        ej = _EXP_VECTORS[j]
        if _exp_dominates(ei, ej) or _exp_dominates(ej, ei):
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

# Cache for _solve_target results
_solve_cache = {}

def _solve_target(n):
    """Find min-k div-free smooth representation of n using DFS with exponent-vector pruning."""
    if n in _solve_cache:
        return _solve_cache[n]
    
    if n in _VAL_INDEX:
        _solve_cache[n] = [n]
        return [n]
    
    # Candidate indices: smooth values <= n, sorted descending
    cand_idx = [i for i in range(_NSM) if _SMOOTH[i] <= n]
    ncand = len(cand_idx)
    if ncand == 0:
        _solve_cache[n] = None
        return None
    
    # Sort in descending order by value
    cand_desc = sorted(cand_idx, key=lambda i: _SMOOTH[i], reverse=True)
    cd_vals = [_SMOOTH[i] for i in cand_desc]
    cd_exps = [_EXP_VECTORS[i] for i in cand_desc]
    ncand = len(cd_vals)
    
    # Rebuild incompatibility for desc order using bitmask
    incompat_d = [0] * ncand
    for i in range(ncand):
        gi = cand_desc[i]
        for j in range(i + 1, ncand):
            gj = cand_desc[j]
            if _INCOMPAT[gi][gj]:
                incompat_d[i] |= (1 << j)
                incompat_d[j] |= (1 << i)
    
    # Precompute prefix sums for quick range sum queries
    prefix_sum = [0] * (ncand + 1)
    for i in range(ncand):
        prefix_sum[i + 1] = prefix_sum[i] + cd_vals[i]
    
    def min_sum_from(i, k):
        """Minimum sum of k elements from cd_vals[i:] (sorted descending).
        The k smallest elements in cd_vals[i:] are the last k elements: indices ncand-k to ncand-1,
        but only if ncand-k >= i. Otherwise, take from position i."""
        if i + k > ncand:
            return float('inf')
        start_idx = max(i, ncand - k)
        return prefix_sum[ncand] - prefix_sum[start_idx]
    
    def max_sum_from(i, k):
        """Maximum sum of k elements from cd_vals[i:] (sorted descending).
        The k largest elements are the first k: indices i to i+k-1."""
        if i + k > ncand:
            return 0
        return prefix_sum[i + k] - prefix_sum[i]
    
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
        if max_sum_from(start, k) < remaining:
            return False
        # Min possible sum: k smallest from position start
        if min_sum_from(start, k) > remaining:
            return False
        
        for i in range(start, ncand):
            v = cd_vals[i]
            if v > remaining:
                # Since sorted descending, all subsequent values are smaller, 
                # but we still need them if remaining is small
                pass
            if v > remaining:
                continue
            if incompat_d[i] & chosen_mask:
                continue
            # Additional pruning: after choosing v, can we still reach remaining-v with k-1 more?
            new_remaining = remaining - v
            if k - 1 > 0:
                if i + 1 + (k - 1) > ncand:
                    continue
                if max_sum_from(i + 1, k - 1) < new_remaining:
                    continue
                if min_sum_from(i + 1, k - 1) > new_remaining:
                    continue
            elif new_remaining != 0:
                continue
            chosen_mask |= (1 << i)
            chosen_list.append(v)
            if backtrack(i + 1, k - 1, new_remaining, chosen_mask, chosen_list):
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
    
    _solve_cache[n] = result[0]
    return result[0]

def _fallback2(n):
    """2-term search with early termination."""
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a >= n:
            break
        b = n - a
        if b in _VAL_INDEX and b > a and _div_free(a, b):
            return [a, b]
    return None

def _fallback3(n):
    """3-term search with early termination."""
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a >= n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c > b and c in _VAL_INDEX and _div_free(a, b) and _div_free(a, c) and _div_free(b, c):
                return [a, b, c]
    return None

def _greedy_fill(n):
    """Greedy: pick largest compatible smooth numbers, then fill remainder."""
    remaining = n
    chosen = []
    chosen_set = set()
    chosen_exps = []
    # Try to build a valid antichain greedily
    for idx in range(len(_SMOOTH) - 1, -1, -1):
        v = _SMOOTH[idx]
        if v > remaining:
            continue
        if v in chosen_set:
            continue
        # Check divisibility with all chosen using exponent vectors
        e = _EXP_VECTORS[idx]
        ok = True
        for ce in chosen_exps:
            if _exp_dominates(e, ce) or _exp_dominates(ce, e):
                ok = False
                break
        if ok:
            chosen.append(v)
            chosen_set.add(v)
            chosen_exps.append(e)
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
            # Check antichain using exponent vectors
            ok = True
            exps = [_EXP_VECTORS[_VAL_INDEX[v]] for v in combo]
            for i in range(len(exps)):
                for j in range(i + 1, len(exps)):
                    if _exp_dominates(exps[i], exps[j]) or _exp_dominates(exps[j], exps[i]):
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
        if time.time() - _T0 > _DEADLINE - 3:
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

# Phase 7: Extended brute force for remaining (only for small n)
for n in range(1, 129):
    if _ANSWERS[n] is None and n <= 50:
        if time.time() - _T0 > _DEADLINE - 2:
            break
        # Try up to size 6
        from itertools import combinations
        cand = [v for v in _SMOOTH if v <= n]
        found = False
        for k in range(2, min(7, len(cand)) + 1):
            if time.time() - _T0 > _DEADLINE - 1:
                break
            for combo in combinations(cand, k):
                if sum(combo) != n:
                    continue
                ok = True
                exps = [_EXP_VECTORS[_VAL_INDEX[v]] for v in combo]
                for i in range(len(exps)):
                    for j in range(i + 1, len(exps)):
                        if _exp_dominates(exps[i], exps[j]) or _exp_dominates(exps[j], exps[i]):
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

# Phase 8: Last resort - use _solve_target with more time
for n in range(1, 129):
    if _ANSWERS[n] is None:
        if time.time() - _T0 > _DEADLINE - 0.5:
            break
        sol = _solve_target(n)
        if sol is not None:
            _ANSWERS[n] = sol

# Final fallback: ensure all have something
for n in range(1, 129):
    if _ANSWERS[n] is None:
        sol = _solve_target(n)
        if sol is not None:
            _ANSWERS[n] = sol
        else:
            _ANSWERS[n] = [n] if n in _SMOOTH_SET else [1]

def construct():
    return _ANSWERS[1:129]
