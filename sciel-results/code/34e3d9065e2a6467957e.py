import time
from functools import lru_cache

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

_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_N_TARGETS = 128

# Precompute incompatibility bitmask
# incompat[i] is a bitmask of indices j such that _SMOOTH[i] and _SMOOTH[j] are incompatible (one divides the other)
_incompat = [0] * _N
for i in range(_N):
    for j in range(i + 1, _N):
        if _SMOOTH[j] % _SMOOTH[i] == 0:
            _incompat[i] |= (1 << j)
            _incompat[j] |= (1 << i)

# Precompute prefix sums for pruning?
# We need to check if we can form `rem_sum` with `rem_terms` from the available set.
# This is hard to do quickly with bitmasks.
# However, we can use a simple bound:
# Max possible sum with `rem_terms` terms from indices >= `start_idx` compatible with `used_mask`.
# This is expensive to compute dynamically.
# Alternative: Since N is small (~30), we can just iterate.

def _find_minimal_rep(target):
    """Find the minimal number of terms to sum to target, returning the terms."""
    # Try k from 1 to target (or smaller)
    # Max terms is target (all 1s, but distinct required, so max is limited by count of smooths)
    # Actually, distinct terms, so max terms is _N.
    # But sum is target, so max terms is target (if 1s were available, but distinct).
    # Max distinct terms summing to T is when terms are 1, 2, 3... k(k+1)/2 <= T.
    # For T=128, k is around 15.
    
    def dfs(rem_sum, rem_terms, start_idx, used_mask, path):
        if rem_terms == 0:
            return rem_sum == 0
        if rem_sum <= 0:
            return False
        
        # Pruning:
        # 1. If rem_terms > rem_sum, impossible (min term 1)
        if rem_terms > rem_sum:
            return False
            
        # 2. Check if we have enough compatible numbers left
        # Count available numbers from start_idx onwards that are compatible with used_mask
        # and _SMOOTH[i] <= rem_sum
        
        # Simple pruning: 
        # The largest possible sum we can make with rem_terms numbers
        # is picking the largest rem_terms numbers from the available set.
        # The smallest possible sum is picking the smallest rem_terms numbers.
        
        # Let's just iterate
        for i in range(start_idx, _N):
            if _SMOOTH[i] > rem_sum:
                break
            
            # Check compatibility
            if used_mask & (1 << i):
                continue
            if _incompat[i] & used_mask:
                continue
                
            # Add to path
            path.append(_SMOOTH[i])
            if dfs(rem_sum - _SMOOTH[i], rem_terms - 1, i + 1, used_mask | (1 << i), path):
                return True
            path.pop()
            
        return False

    for k in range(1, 20): # Max terms likely small
        if dfs(target, k, 0, 0, []):
            # Retrieve path
            # The dfs above modifies path. We need to return it.
            # Let's rewrite to return the list.
            pass
            
    # Fallback: Greedy
    # This is a placeholder for the actual implementation below
    return []

# The above DFS needs to return the path. Let's implement a version that does.
# Also, we need to cache results.

_OPTIMAL_REPS = {}

def _compute_optimal():
    for t in range(1, _N_TARGETS + 1):
        if t in _OPTIMAL_REPS:
            continue
            
        found = None
        for k in range(1, 20):
            # Use a separate function to find and return the list
            res = _dfs_find(t, k, 0, 0)
            if res is not None:
                found = res
                break
        
        if found is None:
            # Should not happen, but fallback to greedy
            found = _greedy(t)
            
        _OPTIMAL_REPS[t] = found

def _dfs_find(rem_sum, rem_terms, start_idx, used_mask):
    if rem_terms == 0:
        return [] if rem_sum == 0 else None
    if rem_sum <= 0:
        return None
    if rem_terms > rem_sum:
        return None
        
    for i in range(start_idx, _N):
        if _SMOOTH[i] > rem_sum:
            break
        if used_mask & (1 << i):
            continue
        if _incompat[i] & used_mask:
            continue
            
        sub_res = _dfs_find(rem_sum - _SMOOTH[i], rem_terms - 1, i + 1, used_mask | (1 << i))
        if sub_res is not None:
            return [_SMOOTH[i]] + sub_res
    return None

def _greedy(target):
    # Simple fallback
    terms = []
    rem = target
    for v in reversed(_SMOOTH):
        if v > rem:
            continue
        ok = True
        for u in terms:
            if u % v == 0 or v % u == 0:
                ok = False
                break
        if ok:
            terms.append(v)
            rem -= v
            if rem == 0:
                break
    return terms

# Precompute at import time to avoid latency in construct()
_start_time = time.time()
_compute_optimal()
# print(f"Precomputation took {time.time() - _start_time:.2f}s")

def construct():
    results = []
    for t in range(1, _N_TARGETS + 1):
        results.append(_OPTIMAL_REPS[t])
    return results
