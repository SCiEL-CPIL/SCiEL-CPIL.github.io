"""Program for large reciprocal-subsum-free sets using optimized greedy + local search."""
import time
import math
from itertools import combinations
from fractions import Fraction
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = 82.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle for efficiency with pruning.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    total = sum(items)
    if total < target:
        return False
    
    # Sort items to improve pruning
    items_sorted = sorted(items, reverse=True)
    
    # Quick check: if largest item alone equals target
    if items_sorted[0] == target:
        return True
    if items_sorted[0] > target:
        # Remove items larger than target (they can't be in any valid subset)
        items_filtered = [x for x in items_sorted if x <= target]
        if not items_filtered:
            return False
        if sum(items_filtered) < target:
            return False
        items_sorted = items_filtered
        n = len(items_sorted)
    
    mid = n // 2
    left = items_sorted[:mid]
    right = items_sorted[mid:]
    
    # Generate all subset sums for left half, pruning sums > target
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # Generate all subset sums for right half, pruning sums > target
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check if any combination of left and right sums equals target
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False


# Cache for _is_subsum_free results
_subsum_free_cache = {}

def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must NOT equal sum(1/y for y in T)
    where T is a non-empty subset of S \\ {x}.
    Uses caching to avoid redundant computations.
    """
    if len(S) < 2:
        return True
    
    # Use frozenset as cache key
    key = frozenset(S)
    if key in _subsum_free_cache:
        return _subsum_free_cache[key]
    
    S_list = list(S)
    n = len(S_list)
    
    L = 1
    for x in S_list:
        L = _lcm(L, x)
        
    vals = [L // x for x in S_list]
    
    result = True
    for i in range(n):
        target = vals[i]
        others = [v for j, v in enumerate(vals) if j != i]
        
        if _can_form_sum(target, others):
            result = False
            break
            
    _subsum_free_cache[key] = result
    return result


def _try_add_elements(current, candidates, max_add=3):
    """
    Try to add up to max_add elements from candidates to current while maintaining subsum-freeness.
    Returns the best extended set found.
    """
    if not candidates or not current:
        return current
    
    best = set(current)
    best_size = len(best)
    
    # Try adding 1 element
    for c in candidates:
        if _time_remaining() < 3.0:
            break
        new_set = best | {c}
        if _is_subsum_free(new_set):
            if len(new_set) > best_size:
                best = new_set
                best_size = len(new_set)
    
    # Try adding 2 elements
    if len(candidates) >= 2 and _time_remaining() > 5.0:
        for c1, c2 in combinations(candidates, 2):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2}
            if _is_subsum_free(new_set):
                if len(new_set) > best_size:
                    best = new_set
                    best_size = len(new_set)
    
    # Try adding 3 elements
    if len(candidates) >= 3 and _time_remaining() > 8.0:
        for c1, c2, c3 in combinations(candidates, 3):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2, c3}
            if _is_subsum_free(new_set):
                if len(new_set) > best_size:
                    best = new_set
                    best_size = len(new_set)
    
    return best


def _swap_improve(current):
    """
    Try to improve current set by swapping out one element for one or more others.
    Returns the best set found.
    """
    if not current or _time_remaining() < 5.0:
        return current
    
    best = set(current)
    best_size = len(best)
    all_elements = set(range(1, 25))
    missing = all_elements - best
    
    if not missing:
        return best
    
    # Try swapping out one element for one or more missing elements
    for elem_to_remove in list(best):
        if _time_remaining() < 3.0:
            break
        
        base_without = best - {elem_to_remove}
        
        # Try adding one missing element
        for m in missing:
            if _time_remaining() < 2.0:
                break
            new_set = base_without | {m}
            if _is_subsum_free(new_set):
                if len(new_set) > best_size:
                    best = new_set
                    best_size = len(new_set)
        
        # Try adding two missing elements
        if len(missing) >= 2 and _time_remaining() > 5.0:
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 2.0:
                    break
                new_set = base_without | {m1, m2}
                if _is_subsum_free(new_set):
                    if len(new_set) > best_size:
                        best = new_set
                        best_size = len(new_set)
    
    return best


def _greedy_descending():
    """Greedy construction from 24 down to 1."""
    current = set()
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_ascending():
    """Greedy construction from 1 up to 24."""
    current = set()
    for k in range(1, 25):
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _greedy_random(seed_val):
    """Greedy construction with random ordering."""
    import random
    rng = random.Random(seed_val)
    elems = list(range(1, 25))
    rng.shuffle(elems)
    current = set()
    for k in elems:
        if _time_remaining() < 5.0:
            break
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    return current


def _local_search(current):
    """Apply local search improvements."""
    if not current:
        return current
    
    all_elements = set(range(1, 25))
    missing = all_elements - current
    
    # Multiple passes to converge
    for _ in range(3):
        if _time_remaining() < 10.0:
            break
        
        if missing:
            current = _try_add_elements(current, list(missing), max_add=3)
        
        if _time_remaining() < 15.0:
            break
        
        current = _swap_improve(current)
        
        missing = all_elements - current
        if not missing:
            break
        
        if _time_remaining() < 8.0:
            break
        
        current = _try_add_elements(current, list(missing), max_add=2)
        missing = all_elements - current
    
    return current


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    best = set()
    best_size = 0
    
    # Try multiple greedy strategies and keep the best
    strategies = [
        _greedy_descending,
        _greedy_ascending,
    ]
    
    for strategy in strategies:
        if _time_remaining() < 10.0:
            break
        candidate = strategy()
        if len(candidate) > best_size:
            best = set(candidate)
            best_size = len(best)
    
    # Try several random orderings
    for seed in range(10):
        if _time_remaining() < 8.0:
            break
        candidate = _greedy_random(seed)
        if len(candidate) > best_size:
            best = set(candidate)
            best_size = len(best)
    
    if not best:
        return [24]
    
    # Apply local search to the best candidate
    best = _local_search(best)
    best_size = len(best)
    
    # Try local search on other promising candidates if time permits
    all_candidates = []
    if _time_remaining() > 20.0:
        all_candidates.append(_greedy_descending())
        all_candidates.append(_greedy_ascending())
    for seed in range(5):
        if _time_remaining() < 15.0:
            break
        all_candidates.append(_greedy_random(seed))
    
    for candidate in all_candidates:
        if _time_remaining() < 12.0:
            break
        if len(candidate) >= best_size:
            improved = _local_search(candidate)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    return list(best)
