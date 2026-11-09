"""Optimized solver: minimum distinct divisors of 5040 summing to 4979."""
import time
from itertools import combinations

# Precompute divisors of 5040, sorted descending
_DIVS = sorted([d for d in range(1, 5041) if 5040 % d == 0], reverse=True)
_N = len(_DIVS)
_DIVS_SET = set(_DIVS)

# Suffix sums: _SUF[i] = sum of _DIVS[i:]
_SUF = [0] * (_N + 1)
for _i in range(_N - 1, -1, -1):
    _SUF[_i] = _SUF[_i + 1] + _DIVS[_i]

# Prefix sums of largest k from start: _PFX_LARGEST[start][k] = sum of _DIVS[start:start+k]
# For fast max-possible-sum pruning
_N_MAX = 256
_TARGET = 4979
_DEADLINE = 85.0


def _sum_largest(start, k):
    """Sum of k largest divisors from index start onward."""
    if start + k > _N:
        return 0
    return _SUF[start] - _SUF[start + k]


def _sum_smallest(start, k):
    """Sum of k smallest divisors from index start onward (i.e., from the end)."""
    # The k smallest available from start.._N-1 are the last k elements
    end = _N
    begin = max(start, _N - k)
    return _SUF[begin] - _SUF[end]


def construct():
    start_time = time.time()
    
    # Quick greedy upper bound
    rem = _TARGET
    greedy = []
    for d in _DIVS:
        if d <= rem:
            greedy.append(d)
            rem -= d
            if rem == 0:
                break
    upper_k = len(greedy) if rem == 0 else _N_MAX
    
    # Try k = 1, 2, 3, ... in order
    for k in range(1, upper_k + 1):
        if time.time() - start_time > _DEADLINE:
            break
        
        # Feasibility checks
        if _sum_largest(0, k) < _TARGET:
            continue
        if _sum_smallest(0, k) > _TARGET:
            continue
        
        # For small k, use specialized enumeration
        if k == 1:
            if _TARGET in _DIVS_SET:
                return [_TARGET]
            continue
        
        if k == 2:
            # Find two distinct divisors summing to target
            for i in range(_N):
                d1 = _DIVS[i]
                d2 = _TARGET - d1
                if d2 <= d1:  # d2 < d1 to ensure distinct and avoid duplicates
                    break
                if d2 in _DIVS_SET:
                    # Verify d2 is at a different index (it will be since d2 < d1 and sorted desc)
                    return [d1, d2]
            continue
        
        if k == 3:
            # Find three distinct divisors summing to target
            # Use two-pointer approach on sorted divisors
            found = False
            for i in range(_N - 2):
                d1 = _DIVS[i]
                if d1 >= _TARGET:
                    continue
                # Need d2 + d3 = _TARGET - d1, with d2 > d3 > 0, all distinct divisors
                # d2 < d1 (since sorted descending, indices i < j < l means d1 > d2 > d3)
                rem23 = _TARGET - d1
                # Two-pointer on _DIVS[i+1:]
                lo = i + 1
                hi = _N - 1
                while lo < hi:
                    s = _DIVS[lo] + _DIVS[hi]
                    if s == rem23:
                        return [d1, _DIVS[lo], _DIVS[hi]]
                    elif s < rem23:
                        hi -= 1
                    else:
                        lo += 1
            continue
        
        # For k >= 4, use DFS with strong pruning
        found = [False]
        path = []
        
        def dfs(k_rem, start_idx, cur_sum):
            if found[0]:
                return
            if time.time() - start_time > _DEADLINE:
                return
            if k_rem == 0:
                if cur_sum == _TARGET:
                    found[0] = True
                return
            if cur_sum > _TARGET:
                return
            if _N - start_idx < k_rem:
                return
            # Max possible: take k_rem largest from start_idx
            if cur_sum + _sum_largest(start_idx, k_rem) < _TARGET:
                return
            # Min possible: take k_rem smallest from start_idx
            if cur_sum + _sum_smallest(start_idx, k_rem) > _TARGET:
                return
            
            for i in range(start_idx, _N):
                d = _DIVS[i]
                if d > _TARGET - cur_sum:
                    continue
                # Prune: even if we take the k_rem-1 largest after i, can we reach target?
                if cur_sum + d + _sum_largest(i + 1, k_rem - 1) < _TARGET:
                    break  # Since _DIVS is sorted desc, further i will be smaller
                path.append(d)
                dfs(k_rem - 1, i + 1, cur_sum + d)
                path.pop()
                if found[0]:
                    return
                if time.time() - start_time > _DEADLINE:
                    return
        
        dfs(k, 0, 0)
        if found[0]:
            return list(path)
    
    # Fallback: return greedy
    return greedy


if __name__ == "__main__":
    result = construct()
    print(f"Count: {len(result)}, Sum: {sum(result)}")
    print(result)
    assert sum(result) == 4979
    assert all(5040 % d == 0 for d in result)
    assert len(set(result)) == len(result)
