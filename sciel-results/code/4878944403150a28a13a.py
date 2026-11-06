"""Rewritten program: meet-in-the-middle for small k, fast greedy+local search for large k."""
import time
import sys

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

# Precompute prefix sums for pruning
_divs_sorted = _divs  # already sorted ascending
_prefix_sum = [0] * (_n_divs + 1)
for i in range(_n_divs):
    _prefix_sum[i + 1] = _prefix_sum[i] + _divs_sorted[i]

def _can_reach(rem, start_idx, k):
    """Can we pick k elements from _divs_sorted[start_idx:] that sum to rem?"""
    if k == 0:
        return rem == 0
    if k < 0:
        return False
    # Max possible sum from start_idx with k elements
    if start_idx + k > _n_divs:
        return False
    max_sum = _prefix_sum[_n_divs] - _prefix_sum[_n_divs - k]
    if max_sum < rem:
        return False
    # Min possible sum from start_idx with k elements
    min_sum = 0
    for i in range(start_idx, min(start_idx + k, _n_divs)):
        min_sum += _divs_sorted[i]
    if min_sum > rem:
        return False
    return True

def _search_k(rem, start_idx, k, chosen):
    """Find k elements from _divs_sorted[start_idx:] summing to rem."""
    if k == 0:
        if rem == 0:
            return chosen
        return None
    if start_idx >= _n_divs:
        return None
    # Pruning
    if start_idx + k > _n_divs:
        return None
    # Max sum check
    max_sum = _prefix_sum[_n_divs] - _prefix_sum[_n_divs - k]
    if max_sum < rem:
        return None
    # Min sum check
    min_sum = 0
    for i in range(start_idx, min(start_idx + k, _n_divs)):
        min_sum += _divs_sorted[i]
    if min_sum > rem:
        return None
    
    for i in range(start_idx, _n_divs - k + 1):
        d = _divs_sorted[i]
        if d > rem:
            break
        # Check if remaining can be reached
        new_rem = rem - d
        new_k = k - 1
        if new_k > 0:
            if i + 1 + new_k > _n_divs:
                break
            # Quick min/max check for remaining
            if i + 1 + new_k <= _n_divs:
                max_rem = _prefix_sum[_n_divs] - _prefix_sum[_n_divs - new_k]
                if max_rem < new_rem:
                    break
                min_rem = 0
                for j in range(i + 1, min(i + 1 + new_k, _n_divs)):
                    min_rem += _divs_sorted[j]
                if min_rem > new_rem:
                    continue
        chosen.append(d)
        result = _search_k(new_rem, i + 1, new_k, chosen)
        if result is not None:
            return result
        chosen.pop()
    return None

def _meet_in_middle_k(k):
    """Meet-in-the-middle for exactly k divisors summing to target."""
    if k <= 1:
        if k == 0:
            return [] if _target == 0 else None
        for d in _divs:
            if d == _target:
                return [d]
        return None
    
    if k == 2:
        for i in range(_n_divs):
            complement = _target - _divs[i]
            if complement > _divs[i]:
                # Binary search
                lo, hi = i + 1, _n_divs - 1
                found = False
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if _divs[mid] == complement:
                        return [_divs[i], _divs[mid]]
                    elif _divs[mid] < complement:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        return None
    
    if k == 3:
        # Two-sum style: for each pair, check if third exists
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                s2 = _divs[i] + _divs[j]
                if s2 >= _target:
                    break
                complement = _target - s2
                if complement <= _divs[j]:
                    continue
                # Binary search for complement in divs[j+1:]
                lo, hi = j + 1, _n_divs - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if _divs[mid] == complement:
                        return [_divs[i], _divs[j], _divs[mid]]
                    elif _divs[mid] < complement:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        return None
    
    if k == 4:
        # Meet in middle: split into 2+2
        # Build all 2-sums with indices
        from collections import defaultdict
        pair_sums = defaultdict(list)
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                s = _divs[i] + _divs[j]
                if s < _target:
                    pair_sums[s].append((i, j))
        
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                s1 = _divs[i] + _divs[j]
                if s1 >= _target:
                    break
                complement = _target - s1
                if complement in pair_sums:
                    for (k1, k2) in pair_sums[complement]:
                        if k1 > j:  # ensure all indices distinct and in order
                            return [_divs[i], _divs[j], _divs[k1], _divs[k2]]
        return None
    
    if k == 5:
        # Split into 2+3
        from collections import defaultdict
        pair_sums = defaultdict(list)
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                s = _divs[i] + _divs[j]
                if s < _target:
                    pair_sums[s].append((i, j))
        
        triple_sums = defaultdict(list)
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                for m in range(j + 1, _n_divs):
                    s = _divs[i] + _divs[j] + _divs[m]
                    if s < _target:
                        triple_sums[s].append((i, j, m))
        
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                s1 = _divs[i] + _divs[j]
                if s1 >= _target:
                    break
                complement = _target - s1
                if complement in triple_sums:
                    for (k1, k2, k3) in triple_sums[complement]:
                        if k1 > j:
                            return [_divs[i], _divs[j], _divs[k1], _divs[k2], _divs[k3]]
        return None
    
    if k == 6:
        # Split into 3+3
        from collections import defaultdict
        triple_sums = defaultdict(list)
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                for m in range(j + 1, _n_divs):
                    s = _divs[i] + _divs[j] + _divs[m]
                    if s < _target:
                        triple_sums[s].append((i, j, m))
        
        for i in range(_n_divs):
            for j in range(i + 1, _n_divs):
                for m in range(j + 1, _n_divs):
                    s1 = _divs[i] + _divs[j] + _divs[m]
                    if s1 >= _target:
                        break
                    complement = _target - s1
                    if complement in triple_sums:
                        for (p, q, r) in triple_sums[complement]:
                            if p > m:
                                return [_divs[i], _divs[j], _divs[m], _divs[p], _divs[q], _divs[r]]
        return None
    
    # For k >= 7, use recursive search with strong pruning
    chosen = []
    return _search_k(_target, 0, k, chosen)

def _greedy_solve(rem, available, max_k=256):
    """Greedy: pick largest available divisor <= rem repeatedly."""
    chosen = []
    remaining = list(available)
    remaining.sort(reverse=True)
    cur = rem
    for d in remaining:
        if d <= cur:
            chosen.append(d)
            cur -= d
            if cur == 0:
                return chosen
    return None

def _local_search_improve(initial, deadline):
    """Try to reduce the number of terms via local search."""
    if initial is None:
        return None
    best = list(initial)
    best_sum = sum(best)
    if best_sum != _target:
        return None
    
    best_k = len(best)
    best_set = set(best)
    
    # Try to find a solution with fewer terms
    for target_k in range(1, best_k):
        if time.time() > deadline:
            break
        result = _meet_in_middle_k(target_k)
        if result is not None:
            return result
    
    return best

def construct():
    deadline = time.time() + 55.0
    target = _target
    
    # Try increasing k from 1 upward
    for k in range(1, min(257, _n_divs + 1)):
        if time.time() > deadline:
            break
        
        result = _meet_in_middle_k(k)
        if result is not None:
            # Verify
            assert len(result) == k
            assert sum(result) == target
            assert len(set(result)) == len(result)
            assert all(d in _divs for d in result)
            return result
    
    # Fallback: greedy from largest
    remainder = target
    chosen = []
    for d in sorted(_divs, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    if remainder == 0:
        return chosen
    
    # Last resort: return what we have (should not happen for this instance)
    return chosen
