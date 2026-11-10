"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_PAIRS = list(combinations(range(N), 2))
TOTAL_PAIRS = len(ALL_PAIRS)  # 91

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products) over edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _evaluate(values, edges):
    """Return the score for a given assignment."""
    return _score(values, edges)

def _random_values():
    """Generate 14 distinct positive integers <= 10000."""
    vals = random.sample(range(1, MAX_VAL + 1), N)
    return vals

def _greedy_edges(values, k_target=None):
    """
    Greedily select M edges to minimize the number of distinct sums/products.
    We try to reuse existing sum/product values as much as possible.
    """
    sums = set()
    prods = set()
    edges = []
    remaining = list(ALL_PAIRS)
    random.shuffle(remaining)
    
    for i, j in remaining:
        if len(edges) >= M:
            break
        s = values[i] + values[j]
        p = values[i] * values[j]
        # Prefer edges that reuse existing sums or products
        if s in sums or p in prods:
            edges.append([i, j])
            sums.add(s)
            prods.add(p)
    
    # If we didn't get enough edges, add more (least costly first)
    if len(edges) < M:
        used = set(tuple(e) for e in edges)
        for i, j in remaining:
            if len(edges) >= M:
                break
            if (i, j) not in used:
                edges.append([i, j])
                sums.add(values[i] + values[j])
                prods.add(values[i] * values[j])
    
    return edges[:M]

def _local_search_values(values, edges, max_iter=5000, deadline=None):
    """
    Local search on values to reduce the number of distinct sums/products.
    We try changing individual values to reduce the objective.
    """
    current_score = _score(values, edges)
    best_values = list(values)
    best_score = current_score
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random vertex to modify
        v = random.randint(0, N - 1)
        old_val = values[v]
        
        # Try a few candidate new values
        improved = False
        for _ in range(10):
            if deadline is not None and time.time() > deadline:
                break
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in values and values.index(new_val) != v:
                continue
            
            values[v] = new_val
            new_score = _score(values, edges)
            if new_score < best_score:
                best_score = new_score
                best_values = list(values)
                improved = True
                break
            values[v] = old_val
        
        if not improved:
            # Occasionally try a random restart of a few values
            for _ in range(3):
                v2 = random.randint(0, N - 1)
                old_v2 = values[v2]
                for _ in range(5):
                    if deadline is not None and time.time() > deadline:
                        break
                    new_v2 = random.randint(1, MAX_VAL)
                    if new_v2 == old_v2:
                        continue
                    if new_v2 in values and values.index(new_v2) != v2:
                        continue
                    values[v2] = new_v2
                    new_score = _score(values, edges)
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                        break
                    values[v2] = old_v2
    
    values[:] = best_values
    return best_score

def _local_search_edges(values, edges, max_iter=5000, deadline=None):
    """
    Local search on edges: try swapping an edge with a non-edge to reduce distinct sums/products.
    """
    current_score = _score(values, edges)
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    edge_set = set(tuple(e) for e in edges)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to potentially remove
        idx = random.randint(0, M - 1)
        removed_edge = tuple(edges[idx])
        
        # Find a non-edge to add
        added_edge = None
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            cand = ALL_PAIRS[random.randint(0, TOTAL_PAIRS - 1)]
            if cand not in edge_set:
                added_edge = cand
                break
        
        if added_edge is None:
            continue
        
        # Try the swap
        new_edges = [list(e) for e in edges if tuple(e) != removed_edge]
        new_edges.append(list(added_edge))
        new_score = _score(values, new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            edge_set = set(tuple(e) for e in best_edges)
    
    edges[:] = best_edges
    return best_score

def _combined_local_search(values, edges, max_iter=20000, deadline=None):
    """
    Alternate between local search on values and edges.
    """
    best_score = _score(values, edges)
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if iteration % 2 == 0:
            # Search on values
            _local_search_values(values, edges, max_iter=500, deadline=deadline)
        else:
            # Search on edges
            _local_search_edges(values, edges, max_iter=500, deadline=deadline)
        
        current_score = _score(values, edges)
        if current_score < best_score:
            best_score = current_score
            best_values = list(values)
            best_edges = [list(e) for e in edges]
    
    values[:] = best_values
    edges[:] = best_edges
    return best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_score = float('inf')
    best_result = None
    
    # Initial random tries
    n_initial = 20
    for i in range(n_initial):
        if time.time() > deadline:
            break
        
        values = _random_values()
        edges = _greedy_edges(values)
        score = _score(values, edges)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
        
        # Quick local search
        _combined_local_search(values, edges, max_iter=2000, deadline=deadline)
        score = _score(values, edges)
        
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
    
    # If we have time, do more thorough search
    if best_result is not None and time.time() < deadline - 10:
        values, edges = best_result
        _combined_local_search(values, edges, max_iter=50000, deadline=deadline)
        score = _score(values, edges)
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
    
    if best_result is None:
        # Fallback
        values = list(range(1, N + 1))
        edges = [list(e) for e in ALL_PAIRS[:M]]
        best_score = _score(values, edges)
        best_result = (values, edges)
    
    values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
