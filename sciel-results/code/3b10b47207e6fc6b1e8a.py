# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _build_smooth_list(max_val=128):
    return [x for x in range(1, max_val + 1) if _is_smooth(x)]

def _is_antichain(terms):
    """Check that no term divides another."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_min_antichain(n, smooth, max_terms=30):
    """Find antichain representation of n with minimum number of terms using backtracking."""
    candidates = sorted([x for x in smooth if x <= n], reverse=True)
    
    # Try from 1 term up
    for count in range(1, min(max_terms, len(candidates)) + 1):
        # Use backtracking with pruning
        result = _backtrack_antichain(n, candidates, count, [], 0, 0)
        if result is not None:
            return result
    return None

def _backtrack_antichain(target, candidates, count, current, start_idx, remaining):
    """Backtracking search for antichain of exactly 'count' elements summing to target."""
    if len(current) == count:
        if remaining == 0:
            return current
        return None
    
    # Pruning: not enough elements left to fill
    if start_idx + (count - len(current)) > len(candidates):
        return None
    
    # Pruning: even taking the largest available can't reach target
    if remaining < 0:
        return None
    
    for i in range(start_idx, len(candidates)):
        x = candidates[i]
        if x > remaining:
            continue
        
        # Check antichain constraint with existing terms
        ok = True
        for t in current:
            a, b = min(x, t), max(x, t)
            if b % a == 0:
                ok = False
                break
        
        if not ok:
            continue
        
        result = _backtrack_antichain(target, candidates, count, current + [x], i + 1, remaining - x)
        if result is not None:
            return result
    return None

def _precompute():
    """Precompute optimal antichain representations for all targets 1-128."""
    smooth = _build_smooth_list(128)
    smooth_set = set(smooth)
    results = {}
    
    for n in range(1, 129):
        # Try increasing number of terms
        found = None
        for count in range(1, 31):
            candidates = sorted([x for x in smooth if x <= n], reverse=True)
            if len(candidates) < count:
                break
            
            # Special case: count == 1
            if count == 1:
                if n in smooth_set:
                    found = [n]
                    break
                continue
            
            # Special case: count == 2
            if count == 2:
                found = _find_two_term_antichain(n, candidates)
                if found:
                    break
                continue
            
            # For count >= 3, use backtracking
            result = _backtrack_antichain(n, candidates, count, [], 0, n)
            if result is not None:
                found = result
                break
        
        if found is None:
            # Fallback: greedy approach
            found = _greedy_antichain(n, smooth)
        
        results[n] = found
    
    return results

def _find_two_term_antichain(n, candidates):
    """Find a 2-term antichain representation of n."""
    for i in range(len(candidates)):
        a = candidates[i]
        b = n - a
        if b <= 0 or b >= a:
            continue
        if b in set(candidates):
            # Check antichain: a doesn't divide b and b doesn't divide a
            if b % a != 0 and a % b != 0:
                return [a, b]
    return None

def _greedy_antichain(n, smooth):
    """Greedy fallback: pick largest smooth numbers that don't divide each other."""
    remaining = n
    terms = []
    available = sorted([x for x in smooth if x <= n], reverse=True)
    for x in available:
        if x > remaining:
            continue
        # Check antichain constraint with existing terms
        ok = True
        for t in terms:
            a, b = min(x, t), max(x, t)
            if b % a == 0:
                ok = False
                break
        if ok:
            terms.append(x)
            remaining -= x
            if remaining == 0:
                break
    if remaining == 0 and len(terms) >= 1:
        return terms
    # Last resort
    if n in smooth:
        return [n]
    return [1, n-1] if n > 1 else [1]

_PRECOMPUTED = None

def construct():
    global _PRECOMPUTED
    if _PRECOMPUTED is None:
        _PRECOMPUTED = _precompute()
    return _PRECOMPUTED
# EVOLVE-BLOCK-END
