"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edges(values, n, m, rng):
    """Greedily select m edges minimizing max(distinct_sums, distinct_prods)."""
    all_edges = list(combinations(range(n), 2))
    rng.shuffle(all_edges)
    
    selected = []
    sums = set()
    prods = set()
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        
        for e in all_edges:
            if e in selected:
                continue
            u, v = e
            s = values[u] + values[v]
            p = values[u] * values[v]
            
            new_sums = sums | {s}
            new_prods = prods | {p}
            sc = max(len(new_sums), len(new_prods))
            
            if sc < best_score:
                best_score = sc
                best_edge = e
                best_sums = new_sums
                best_prods = new_prods
                if best_score == len(selected) + 1:
                    # Can't do better than adding one new sum and one new product
                    break
        
        if best_edge is None:
            # All remaining edges already considered, pick any unused
            for e in all_edges:
                if e not in selected:
                    best_edge = e
                    break
            if best_edge is None:
                break
        
        selected.append(best_edge)
        sums = best_sums if best_sums is not None else sums | {values[best_edge[0]] + values[best_edge[1]]}
        prods = best_prods if best_prods is not None else prods | {values[best_edge[0]] * values[best_edge[1]]}
    
    return selected

def _local_search(values, edges, n, m, rng, max_iters=500):
    """Try swapping edges to improve score."""
    current_score = _score(values, edges)
    edge_set = set(edges)
    all_edges = list(combinations(range(n), 2))
    
    for _ in range(max_iters):
        # Pick a random edge to remove
        if not edges:
            break
        rem_idx = rng.randrange(len(edges))
        rem_edge = edges[rem_idx]
        
        # Try adding a different edge
        candidates = [e for e in all_edges if e not in edge_set and e != rem_edge]
        if not candidates:
            continue
        
        cand = candidates[rng.randrange(len(candidates))]
        
        new_edges = [e for i, e in enumerate(edges) if i != rem_idx] + [cand]
        new_score = _score(values, new_edges)
        
        if new_score < current_score:
            edges = new_edges
            edge_set = set(edges)
            current_score = new_score
    
    return edges, current_score

def _random_values(n, max_val, rng):
    """Generate random distinct positive integers <= max_val."""
    vals = rng.sample(range(1, max_val + 1), n)
    vals.sort()
    return vals

def _structured_values(n, rng):
    """Generate structured values: start with small consecutive, maybe perturb."""
    base = list(range(1, n + 1))
    # Sometimes use a different base
    r = rng.random()
    if r < 0.3:
        # Consecutive from 1
        return base
    elif r < 0.6:
        # Consecutive from a small offset
        offset = rng.randint(1, 3)
        return [offset + i for i in range(n)]
    else:
        # Random small values
        return rng.sample(range(1, 30), n)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    rng = random.Random(42)
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Try several value configurations
    value_candidates = []
    
    # Base: 1..14
    value_candidates.append(list(range(1, n + 1)))
    # 1..14 with some perturbation
    for _ in range(20):
        vals = list(range(1, n + 1))
        # Swap a couple
        for _ in range(rng.randint(1, 3)):
            i, j = rng.sample(range(n), 2)
            vals[i], vals[j] = vals[j], vals[i]
        value_candidates.append(vals)
    # Random small values
    for _ in range(20):
        value_candidates.append(_random_values(n, 25, rng))
    # Structured
    for _ in range(10):
        value_candidates.append(_structured_values(n, rng))
    
    # Evaluate each candidate
    for vals in value_candidates:
        if time.time() > deadline:
            break
        
        # Ensure distinct
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > max_val for v in vals):
            continue
        
        # Greedy edge selection with multiple random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            edges = _greedy_edges(vals, n, m, rng)
            if len(edges) < m:
                continue
            
            # Local search
            edges, sc = _local_search(vals, edges, n, m, rng, max_iters=200)
            
            if sc < best_score:
                best_score = sc
                best_values = vals[:]
                best_edges = [list(e) for e in edges]
        
        # Early stop if we find a very good solution
        if best_score <= 20:
            break
    
    # If no valid solution found, use fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert i != j
        assert 0 <= i < n and 0 <= j < n
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
