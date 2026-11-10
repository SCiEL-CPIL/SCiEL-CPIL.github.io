"""Optimized program for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        vi = values[i]
        vj = values[j]
        sums.add(vi + vj)
        prods.add(vi * vj)
    return max(len(sums), len(prods))

def _circulant_edges(n=14, steps=None):
    """Generate edges for a circulant graph with given connection steps."""
    if steps is None:
        steps = [1, 2, 3, 4, 5]
    edges = set()
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if j != i:
                a, b = min(i, j), max(i, j)
                edges.add((a, b))
    return [list(e) for e in sorted(edges)]

def _random_edges(n=14, m=35):
    """Generate a random set of m edges on n vertices."""
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    return [list(e) for e in all_edges[:m]]

def _local_search(values, edges, max_iter=50000, timeout=None):
    """Local search: swap values between vertices to reduce score."""
    n = len(values)
    current = list(values)
    best_score = _score(current, edges)
    best_values = list(current)
    
    start = time.time()
    iter_count = 0
    no_improve = 0
    max_no_improve = 3000
    
    while iter_count < max_iter:
        # Check timeout
        if timeout is not None:
            if time.time() - start > timeout:
                break
        
        # Pick two random distinct vertices
        i, j = random.sample(range(n), 2)
        
        # Swap
        current[i], current[j] = current[j], current[i]
        
        # Check constraints: all values must be distinct positive ints <= 10000
        # (swapping preserves distinctness and range, so always valid)
        
        new_score = _score(current, edges)
        iter_count += 1
        
        if new_score < best_score:
            best_score = new_score
            best_values = list(current)
            no_improve = 0
        else:
            # Revert
            current[i], current[j] = current[j], current[i]
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_values, best_score

def _try_value_assignment(edges, seed, max_iter=30000, timeout=5.0):
    """Try to find good values for given edges using local search."""
    rng = random.Random(seed)
    n = P['n']
    start_time = time.time()
    
    # Initial values: try several strategies
    best_overall = None
    best_score = float('inf')
    
    strategies = [
        lambda: list(range(1, n + 1)),  # 1..14
        lambda: [1 + i * 2 for i in range(n)],  # 1,3,5,...,27
        lambda: [1 + i * 7 for i in range(n)],  # 1,8,15,...,92
        lambda: sorted(rng.sample(range(1, 10001), n)),  # random
        lambda: [1 + i * 10 for i in range(n)],  # 1,11,21,...,131
        lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],  # powers of 2
        lambda: [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],  # primes
        lambda: [1, 3, 9, 27, 81, 243, 729, 2187, 3000, 4000, 5000, 6000, 7000, 8000],  # mixed
    ]
    
    for s in range(len(strategies)):
        if timeout is not None and time.time() - start_time > timeout:
            break
        try:
            vals = strategies[s]()
        except:
            continue
        
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
        
        # Distribute remaining time across strategies
        elapsed = time.time() - start_time
        remaining = max(0.1, timeout - elapsed) if timeout else max_iter
        per_strategy = remaining / max(1, len(strategies) - s)
        
        vals, sc = _local_search(vals, edges, max_iter=max_iter, timeout=per_strategy)
        if sc < best_score:
            best_score = sc
            best_overall = list(vals)
    
    return best_overall, best_score

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin
    
    n = P['n']
    m = P['m']
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Try circulant graph first (most promising structure)
    circ_edges = _circulant_edges(n, [1, 2, 3, 4, 5])
    if len(circ_edges) == m:
        vals, sc = _try_value_assignment(circ_edges, seed=42, max_iter=50000, timeout=15.0)
        if vals and sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = circ_edges
    
    # Try other circulant variants with diverse step sets
    circ_step_sets = [
        [1, 2, 3, 5, 6],
        [1, 2, 4, 5, 7],
        [1, 3, 4, 6, 7],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 5, 7],
        [1, 2, 4, 6, 7],
        [1, 3, 4, 5, 7],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 4, 8],
        [1, 2, 3, 5, 8],
    ]
    for steps in circ_step_sets:
        if time.time() > deadline - 10:
            break
        ce = _circulant_edges(n, steps)
        if len(ce) != m:
            continue
        vals, sc = _try_value_assignment(ce, seed=123, max_iter=20000, timeout=6.0)
        if vals and sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = ce
    
    # Try random graphs with different seeds
    for trial in range(8):
        if time.time() > deadline - 5:
            break
        re = _random_edges(n, m)
        vals, sc = _try_value_assignment(re, seed=100 + trial, max_iter=15000, timeout=4.0)
        if vals and sc < best_score:
            best_score = sc
            best_values = vals
            best_edges = re
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_score = _score(best_values, best_edges)
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        a, b = e
        assert isinstance(a, int) and isinstance(b, int)
        assert a != b
        assert 0 <= a < n and 0 <= b < n
        key = (min(a, b), max(a, b))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
# EVOLVE-BLOCK-END
