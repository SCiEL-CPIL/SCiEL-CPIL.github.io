import time
import random
import math
from itertools import combinations

# Deadline guard: 90s limit, use 85s for safety
START_TIME = time.time()
DEADLINE = START_TIME + 85.0

def _check_time():
    return time.time() > DEADLINE

def _score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, n_edges=35, max_iters=10000):
    """
    Find the best set of n_edges edges for a given set of values
    using Greedy Initialization followed by Simulated Annealing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all possible edges
    edge_props = {}
    for e in all_edges:
        u, v = e
        edge_props[e] = (values[u] + values[v], values[u] * values[v])
        
    # Greedy Initialization
    # We want to select edges that add the fewest new distinct sums and products.
    current_edges = set()
    current_sums = set()
    current_prods = set()
    
    # Sort edges by a heuristic: prefer edges with smaller sums and products?
    # Or just iterate and pick the best marginal gain.
    # Since n=14, 91 edges, we can afford O(N^2) greedy steps.
    remaining_edges = set(all_edges)
    
    for _ in range(n_edges):
        best_edge = None
        best_metric = float('inf')
        best_sums = None
        best_prods = None
        
        for e in remaining_edges:
            s, p = edge_props[e]
            # Calculate new sets
            new_sums = current_sums | {s}
            new_prods = current_prods | {p}
            metric = max(len(new_sums), len(new_prods))
            
            # Tie-breaking: prefer smaller sums/products? Or random?
            # Let's add a small penalty for large values to keep numbers small
            # But the metric is the primary goal.
            if metric < best_metric:
                best_metric = metric
                best_edge = e
                best_sums = new_sums
                best_prods = new_prods
        
        if best_edge is not None:
            current_edges.add(best_edge)
            current_sums = best_sums
            current_prods = best_prods
            remaining_edges.remove(best_edge)
        else:
            break
            
    current_score = _score(values, list(current_edges))
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    min_T = 0.5
    
    iter_count = 0
    while T > min_T and iter_count < max_iters:
        if _check_time():
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random edge to add (not currently in set)
        # Efficiently pick from non-edges
        cand_e = None
        attempts = 0
        while cand_e is None or cand_e in current_edges:
            cand_e = random.choice(all_edges)
            attempts += 1
            if attempts > 50:
                break
                
        if cand_e is None or cand_e in current_edges:
            continue
            
        # Calculate new score incrementally?
        # For n=14, full recalc is fast (35 edges).
        new_edges_list = list(current_edges)
        new_edges_list.remove((u, v))
        new_edges_list.append(cand_e)
        
        new_score = _score(values, new_edges_list)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_edges.discard((u, v))
            current_edges.add(cand_e)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
        T *= cooling
        iter_count += 1
        
    return best_score, list(best_edges)

def _generate_candidates():
    """
    Generate value sets with strong multiplicative and additive structure.
    Focus on small integers and highly composite numbers.
    """
    candidates = []
    n = 14
    
    # Strategy 1: Small consecutive integers
    candidates.append(list(range(1, 15)))
    
    # Strategy 2: Highly composite numbers
    hc = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    candidates.append(hc)
    
    # Strategy 3: Powers and products of small primes
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Strategy 4: Dense small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # Strategy 5: Mix of small and medium composites
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 18, 20])
    
    # Strategy 6: Random search over small highly composite numbers
    random.seed(42)
    # Pool of numbers with many divisors in range [1, 50]
    pool = []
    for i in range(1, 51):
        div_count = sum(1 for d in range(1, i+1) if i % d == 0)
        pool.append((div_count, i))
    pool.sort(reverse=True)
    top_pool = [x[1] for x in pool[:30]]
    
    for _ in range(50):
        vals = random.sample(top_pool, 14)
        vals.sort()
        if len(set(vals)) == 14:
            candidates.append(vals)
            
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
            
    return unique_candidates

def construct():
    n = 14
    m = 35
    
    candidates = _generate_candidates()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Iterate through candidates
    # We have ~60 candidates. Each SA run takes some time.
    # Let's limit the SA iterations per candidate to ensure we finish in time.
    # 60 candidates * 2000 iters = 120k iters. Should be fast enough.
    
    for vals in candidates:
        if _check_time():
            break
            
        # Ensure valid values
        if len(set(vals)) != n:
            continue
        if min(vals) < 1 or max(vals) > 10000:
            continue
            
        score, edges = _optimize_edges(vals, n_edges=m, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges[:]
            
            # If we hit a very low score, we can stop early
            if best_score <= 5:
                break
                
    # If we have time, try to improve the best values found
    # by doing a few more SA runs with the best values found
    if best_values is not None and not _check_time():
        # Run SA again with the best values to see if we find a better edge set
        # with a different random seed
        for seed in [1, 2, 3]:
            if _check_time():
                break
            random.seed(seed)
            score, edges = _optimize_edges(best_values, n_edges=m, max_iters=3000)
            if score < best_score:
                best_score = score
                best_edges = edges

    # Final validation
    if best_values is None:
        # Fallback
        best_values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
        best_edges = list(combinations(range(n), 2))[:m]
        
    # Ensure types
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sanity checks
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
