import time
import random
from functools import lru_cache

# Global cache: n -> list of vectors in optimal greedy order
_cache = {}
_start_time = None
_DEADLINE = 310  # Safety margin from 330s kill

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _greedy_size(order, n):
    """Given an ordering of all 3^n vectors, return the cap set size from greedy."""
    added_set = set()
    for v in order:
        # Check if adding v keeps cap set
        # v conflicts with a,b in added_set if a+b+v = 0 mod 3 for some a!=b!=v
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
    return len(added_set)

def _greedy_size_with_set(order, n, added_set):
    """Given an ordering and current added_set, return (new_size, new_added_set) from greedy."""
    for v in order:
        if v in added_set:
            continue
        conflict = False
        for a in added_set:
            b = tuple((-a[i] - v[i]) % 3 for i in range(n))
            if b in added_set and b != a and b != v:
                conflict = True
                break
        if not conflict:
            added_set.add(v)
    return len(added_set), added_set

def _generate_all_vectors(n):
    """Generate all 3^n vectors."""
    if n == 0:
        return [()]
    result = []
    def _gen(prefix, depth):
        if depth == n:
            result.append(tuple(prefix))
            return
        for v in range(3):
            prefix.append(v)
            _gen(prefix, depth+1)
            prefix.pop()
    _gen([], 0)
    return result

def _local_improve(order, n, time_budget=30):
    """Try to improve the order by swapping nearby elements."""
    if _time_left() < time_budget:
        return order, _greedy_size(order, n)
    
    current_size = _greedy_size(order, n)
    best_order = order[:]
    best_size = current_size
    start_time = time.time()
    rng = random.Random(12345)
    
    # Try swaps of nearby elements
    for _ in range(5000):
        if time.time() - start_time > time_budget:
            break
        if _time_left() < 60:
            break
        
        # Pick two nearby positions to swap
        i = rng.randint(0, len(order) - 2)
        j = i + 1 + rng.randint(0, min(20, len(order) - i - 2))
        
        new_order = order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        sz = _greedy_size(new_order, n)
        if sz > best_size:
            best_size = sz
            best_order = new_order
            order = new_order[:]
    
    return best_order, best_size

def _find_best_order(n, time_limit=250):
    """Find a good ordering for greedy cap set construction."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    best_order = None
    best_size = 0
    
    # Start with some structured orderings
    candidates = []
    
    # 1. Weight-based ordering (sum of coordinates)
    def weight(v):
        return sum(v)
    candidates.append(sorted(all_vecs, key=weight, reverse=True))
    candidates.append(sorted(all_vecs, key=weight))
    
    # 2. By number of 1s
    def num_ones(v):
        return sum(1 for x in v if x == 1)
    candidates.append(sorted(all_vecs, key=num_ones, reverse=True))
    
    # 3. By number of 2s
    def num_twos(v):
        return sum(1 for x in v if x == 2)
    candidates.append(sorted(all_vecs, key=num_twos, reverse=True))
    
    # 4. Lexicographic
    candidates.append(sorted(all_vecs))
    candidates.append(sorted(all_vecs, reverse=True))
    
    # 5. Random orderings
    rng = random.Random(42)
    for _ in range(10):
        order = all_vecs[:]
        rng.shuffle(order)
        candidates.append(order)
    
    # Evaluate candidates
    for order in candidates:
        if _time_left() < 60:
            break
        sz = _greedy_size(order, n)
        if sz > best_size:
            best_size = sz
            best_order = order[:]
    
    # Local improvement
    if best_order is not None and _time_left() > 120:
        best_order, best_size = _local_improve(best_order, n, time_budget=60)
    
    # More random restarts with local improvement if time allows
    if _time_left() > 150:
        for _ in range(50):
            if _time_left() < 90:
                break
            order = all_vecs[:]
            rng.shuffle(order)
            sz = _greedy_size(order, n)
            if sz > best_size:
                best_size = sz
                best_order = order[:]
                # Try local improvement on this better order
                if _time_left() > 100:
                    best_order, best_size = _local_improve(best_order, n, time_budget=30)
    
    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=250)
    
    order = _cache[n]
    # Find the index of this vector in the order
    # This is O(3^n) per call which is bad. Use a dict.
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        idx = {v: i for i, v in enumerate(order)}
        priority._index_cache[n] = idx
    
    idx_map = priority._index_cache[n]
    # Higher priority = earlier in order
    return float(len(order) - idx_map[vector])
