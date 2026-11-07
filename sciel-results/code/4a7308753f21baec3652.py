"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        """Fast evaluation using numpy."""
        vals = np.array(values, dtype=np.int64)
        u = np.array([e[0] for e in edges], dtype=np.int64)
        v = np.array([e[1] for e in edges], dtype=np.int64)
        sums = vals[u] + vals[v]
        prods = vals[u] * vals[v]
        n_sums = len(np.unique(sums))
        n_prods = len(np.unique(prods))
        return max(n_sums, n_prods)
    
    def greedy_edges(values, all_edges, rng):
        """Greedy: pick edges that reuse existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = e
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
            edges.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return edges
    
    def local_search(values, edges, rng, max_iter=5000):
        """Local search with edge swaps, value swaps, and value perturbations."""
        all_edges = list(combinations(range(n), 2))
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        best_score = evaluate(best_values, best_edges)
        edge_set = set(tuple(e) for e in best_edges)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try swapping an edge
                idx = rng.randrange(len(best_edges))
                old_edge = tuple(best_edges[idx])
                candidates = [e for e in all_edges if e not in edge_set]
                if not candidates:
                    continue
                new_edge = rng.choice(candidates)
                new_edges = [list(e) for e in best_edges]
                new_edges[idx] = list(new_edge)
                new_score = evaluate(best_values, new_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    edge_set = set(tuple(e) for e in best_edges)
                    
            elif move_type < 0.7:
                # Try swapping two values
                i, j = rng.sample(range(n), 2)
                new_values = list(best_values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                if len(set(new_values)) != n:
                    continue
                new_score = evaluate(new_values, best_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_values = new_values
                    
            else:
                # Try perturbing one value
                i = rng.randrange(n)
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                new_val = best_values[i] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                new_values = list(best_values)
                new_values[i] = new_val
                if len(set(new_values)) != n:
                    continue
                new_score = evaluate(new_values, best_edges)
                if new_score < best_score:
                    best_score = new_score
                    best_values = new_values
        
        return best_values, best_edges, best_score
    
    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_overall = float('inf')
    best_result = None
    
    # Generate diverse value sets
    value_sets = []
    
    # Strategy 1: Small consecutive integers
    for start in range(1, 30):
        vals = list(range(start, start + n))
        if max(vals) <= max_val:
            value_sets.append(vals)
    
    # Strategy 2: Arithmetic progressions with small step
    for start in range(1, 60):
        for step in range(1, 4):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
    
    # Strategy 3: Random values in small range (clustered)
    for _ in range(200):
        lo = rng.randint(1, 100)
        hi = lo + rng.randint(n, 40)
        vals = rng.sample(range(lo, hi + 1), n)
        value_sets.append(vals)
    
    # Strategy 4: Powers and products
    for base in [2, 3]:
        vals = [base ** i for i in range(n)]
        if max(vals) <= max_val:
            value_sets.append(vals)
    
    # Strategy 5: Semi-structured - mix of small and medium values
    for _ in range(100):
        small = rng.sample(range(1, 20), 7)
        medium = rng.sample(range(20, 80), 7)
        vals = small + medium
        if len(set(vals)) == n and max(vals) <= max_val:
            value_sets.append(vals)
    
    # Strategy 6: Values with common factors (to reduce product diversity)
    for base in [2, 3, 5, 6, 7, 10]:
        for _ in range(5):
            multipliers = rng.sample(range(1, 200), n)
            vals = [base * m for m in multipliers]
            if len(set(vals)) == n and max(vals) <= max_val and min(vals) >= 1:
                value_sets.append(vals)
    
    # Deduplicate
    seen_sets = set()
    unique_value_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
    
    # Sort by initial greedy score to prioritize promising sets
    scored_sets = []
    for vs in unique_value_sets:
        if time.time() > deadline - 15:
            break
        edges = greedy_edges(vs, all_edges, rng)
        score = evaluate(vs, edges)
        scored_sets.append((score, vs, edges))
    
    scored_sets.sort(key=lambda x: x[0])
    
    # Evaluate top candidates with local search
    for score, vs, edges in scored_sets:
        if time.time() > deadline - 5:
            break
        if score >= best_overall:
            continue  # Skip if already worse than best
        vals, eds, ls_score = local_search(vs, edges, rng, max_iter=3000)
        if ls_score < best_overall:
            best_overall = ls_score
            best_result = (list(vals), [list(e) for e in eds])
    
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_edges[:m]]
    else:
        vals, edges = best_result
    
    # Final validation
    assert len(vals) == n
    assert len(set(vals)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in vals)
    assert len(edges) == m
    assert all(isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n and u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': vals, 'edges': edges}
# EVOLVE-BLOCK-END
