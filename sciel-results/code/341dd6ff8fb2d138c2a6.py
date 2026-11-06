"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global cache: n -> list of vectors in optimal greedy order
_cache = {}
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

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

def _greedy_size(order, n):
    """Given an ordering of all 3^n vectors, return the cap set size from greedy."""
    S = set()
    count = 0
    # Local variables for speed
    range_n = range(n)
    for v in order:
        conflict = False
        for u in S:
            # Calculate w such that u + v + w = 0 mod 3
            # w = -u - v mod 3
            # Optimization: pre-compute negation?
            # tuple((-u[i] - v[i]) % 3 for i in range_n)
            w = tuple((-u[i] - v[i]) % 3 for i in range_n)
            if w in S and w != u:
                conflict = True
                break
        if not conflict:
            S.add(v)
            count += 1
    return count

def _find_best_order(n, time_limit=300):
    """Find a good ordering for greedy cap set construction using Simulated Annealing."""
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    total = len(all_vecs)
    
    # Initial Candidates
    candidates = []
    
    # 1. Weight based sorts
    def weight_sum(v):
        return sum(v)
    def weight_nonzero(v):
        return sum(1 for x in v if x != 0)
    def weight_ones(v):
        return sum(1 for x in v if x == 1)
        
    candidates.append(sorted(all_vecs, key=weight_sum, reverse=True))
    candidates.append(sorted(all_vecs, key=weight_sum))
    candidates.append(sorted(all_vecs, key=weight_nonzero, reverse=True))
    candidates.append(sorted(all_vecs, key=weight_ones, reverse=True))
    
    # 2. Random shuffles
    rng = random.Random(42)
    for _ in range(10):
        o = all_vecs[:]
        rng.shuffle(o)
        candidates.append(o)
        
    best_order = None
    best_size = 0
    
    # Evaluate initial candidates
    for o in candidates:
        sz = _greedy_size(o, n)
        if sz > best_size:
            best_size = sz
            best_order = o[:]
            
    if _elapsed() > time_limit - 30:
        _cache[n] = best_order
        return best_order

    # Simulated Annealing
    current_order = best_order[:]
    current_size = best_size
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    steps = 0
    max_time = time_limit - 20 # Safety margin
    
    while _elapsed() < max_time:
        # Pick two random indices to swap
        i = rng.randint(0, total - 1)
        j = rng.randint(0, total - 1)
        if i == j:
            continue
            
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate
        new_size = _greedy_size(current_order, n)
        
        delta = new_size - current_size
        
        if delta > 0:
            # Accept
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = current_order[:]
        else:
            # Metropolis criterion
            if T > 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    # Revert
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        steps += 1
        # Optional: Restart if stuck for too long?
        # For now, simple SA.
        
    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=300)
    
    order = _cache[n]
    
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        idx = {v: i for i, v in enumerate(order)}
        priority._index_cache[n] = idx
    
    idx_map = priority._index_cache[n]
    # Higher priority = earlier in order
    return float(len(order) - idx_map[vector])
# EVOLVE-BLOCK-END
