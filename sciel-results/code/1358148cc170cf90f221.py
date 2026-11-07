"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 85.0

def _evaluate(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _local_search_edges(values, m, max_iters=5000, deadline=None):
    """Local search: start with best initial edges, then swap to improve."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        i, j = e
        edge_sums[e] = values[i] + values[j]
        edge_prods[e] = values[i] * values[j]
    
    # Initial: sort by product ascending, then sum ascending, take first m
    sorted_edges = sorted(all_edges, key=lambda e: (edge_prods[e], edge_sums[e]))
    initial = set(sorted_edges[:m])
    
    def _score(edge_set):
        sums = set()
        prods = set()
        for e in edge_set:
            sums.add(edge_sums[e])
            prods.add(edge_prods[e])
        return max(len(sums), len(prods))
    
    current = initial.copy()
    best_score = _score(current)
    best_set = current.copy()
    
    rng = random.Random(42)
    iters = 0
    while iters < max_iters:
        if deadline and time.time() > deadline:
            break
        # Pick a random edge in current, random edge not in current
        in_list = list(current)
        out_list = [e for e in all_edges if e not in current]
        if not out_list:
            break
        e_in = in_list[rng.randrange(len(in_list))]
        e_out = out_list[rng.randrange(len(out_list))]
        
        new_set = current.copy()
        new_set.remove(e_in)
        new_set.add(e_out)
        s = _score(new_set)
        if s < best_score:
            best_score = s
            best_set = new_set
            current = new_set
        iters += 1
    
    return list(best_set), best_score

def _try_value_set(values, m, deadline=None):
    """Try a specific value set with local search edge selection."""
    edges, score = _local_search_edges(values, m, max_iters=3000, deadline=deadline)
    if len(edges) < m:
        return None, None
    return score, edges

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Value sets with strong collision structure
    candidates = [
        # Dense small integers - good sum collisions
        list(range(1, 15)),
        # Multiplicative structure
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Mix
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27],
        # More multiplicative
        [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 27, 32, 36, 48],
        # Arithmetic progression
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Another dense set
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Geometric-ish
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        # Highly composite
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        # Small with gaps for product collisions
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27],
        # Another variant
        [1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36],
    ]
    
    best_score = float('inf')
    best_result = None
    
    for ci, values in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        deadline_for_this = min(start_time + DEADLINE - 2, time.time() + 7.0)
        score, edges = _try_value_set(values, m, deadline=deadline_for_this)
        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
    
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
