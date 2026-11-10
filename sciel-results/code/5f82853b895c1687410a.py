"""Rewritten program: direct search over value sets and edge subsets to minimize max(distinct sums, distinct products)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Return (raw_value, n_sums, n_prods)."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return (max(len(sums), len(prods)), len(sums), len(prods))

def _all_edges(n):
    return [list(e) for e in combinations(range(n), 2)]

def _greedy_edge_selection(values, n, m):
    """Greedy: pick edges one by one that minimize increase in max(distinct sums, distinct products)."""
    all_e = _all_edges(n)
    chosen = []
    sums = set()
    prods = set()
    # Precompute sum and product for each edge
    edge_data = [(i, j, values[i] + values[j], values[i] * values[j]) for i, j in all_e]
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        for i, j, s, p in edge_data:
            if [i, j] in chosen or [j, i] in chosen:
                continue
            new_sums = sums | {s}
            new_prods = prods | {p}
            sc = max(len(new_sums), len(new_prods))
            if sc < best_score:
                best_score = sc
                best_edge = (i, j)
                best_sums = new_sums
                best_prods = new_prods
        if best_edge is None:
            break
        chosen.append(list(best_edge))
        sums = best_sums
        prods = best_prods
    return chosen

def _local_search_values(values, edges, max_iter=200, seed=0):
    """Local search on values: perturb a value, keep if score improves."""
    rng = random.Random(seed)
    n = len(values)
    best_values = list(values)
    best_score, _, _ = _score(best_values, edges)
    
    for _ in range(max_iter):
        i = rng.randrange(n)
        old = best_values[i]
        # Try a random new value
        new_val = rng.randint(1, P['maximum'])
        if new_val in best_values:
            continue
        best_values[i] = new_val
        # Check distinctness
        if len(set(best_values)) != n:
            best_values[i] = old
            continue
        sc, _, _ = _score(best_values, edges)
        if sc <= best_score:
            best_score = sc
        else:
            best_values[i] = old
    
    return best_values, best_score

def _local_search_edges(values, edges, max_iter=500, seed=0):
    """Local search on edge set: swap one edge for another."""
    rng = random.Random(seed)
    n = len(values)
    all_e = _all_edges(n)
    best_edges = [list(e) for e in edges]
    best_score, _, _ = _score(values, best_edges)
    
    for _ in range(max_iter):
        # Pick a random edge in current set
        idx = rng.randrange(len(best_edges))
        old_edge = best_edges[idx]
        # Pick a random edge not in current set
        new_edge = rng.choice(all_e)
        if new_edge in best_edges:
            continue
        best_edges[idx] = new_edge
        sc, _, _ = _score(values, best_edges)
        if sc <= best_score:
            best_score = sc
        else:
            best_edges[idx] = old_edge
    
    return best_edges, best_score

def construct():
    deadline = time.time() + 85  # 5s safety margin
    n = P['n']
    m = P['m']
    
    best_overall = None
    best_overall_score = float('inf')
    
    # Phase 1: Try several random value sets with greedy edge selection
    base_seed = 42
    rng = random.Random(base_seed)
    
    num_value_trials = 50
    trial = 0
    while trial < num_value_trials and time.time() < deadline:
        # Generate a random set of n distinct values
        values = list(rng.sample(range(1, P['maximum'] + 1), n))
        rng.shuffle(values)
        
        # Greedy edge selection
        edges = _greedy_edge_selection(values, n, m)
        if len(edges) != m:
            trial += 1
            continue
        
        score, ns, np_ = _score(values, edges)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = (list(values), [list(e) for e in edges])
        
        # Local search on values
        if time.time() < deadline:
            vals_ls, sc_ls = _local_search_values(values, edges, max_iter=100, seed=trial)
            if sc_ls < best_overall_score:
                best_overall_score = sc_ls
                best_overall = (vals_ls, [list(e) for e in edges])
        
        trial += 1
    
    # Phase 2: Local search on edges for the best solution found
    if best_overall is not None and time.time() < deadline:
        vals, edges = best_overall
        edges_ls, sc_ls = _local_search_edges(vals, edges, max_iter=300, seed=0)
        if sc_ls < best_overall_score:
            best_overall_score = sc_ls
            best_overall = (vals, edges_ls)
    
    # Phase 3: More value trials with different strategies
    if best_overall is not None and time.time() < deadline:
        for extra_trial in range(30):
            if time.time() >= deadline:
                break
            # Try values with structure: e.g., powers, primes, etc.
            strategy = extra_trial % 3
            if strategy == 0:
                # Random
                values = list(rng.sample(range(1, P['maximum'] + 1), n))
            elif strategy == 1:
                # Small values
                values = list(rng.sample(range(1, 200), n))
            else:
                # Spread out
                values = list(rng.sample(range(100, P['maximum'] + 1), n))
            
            edges = _greedy_edge_selection(values, n, m)
            if len(edges) != m:
                continue
            
            score, ns, np_ = _score(values, edges)
            if score < best_overall_score:
                best_overall_score = score
                best_overall = (list(values), [list(e) for e in edges])
    
    # Fallback
    if best_overall is None:
        values = list(range(1, n + 1))
        edges = _greedy_edge_selection(values, n, m)
        best_overall = (values, edges)
    
    values, edges = best_overall
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
