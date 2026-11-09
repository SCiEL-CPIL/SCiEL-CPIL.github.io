"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85  # 5s safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _greedy_edges(values, m, n, max_iter=5000):
    """Greedy edge selection to minimize distinct sums/products."""
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    edges = []
    sums = set()
    prods = set()
    
    for e in all_edges:
        if len(edges) >= m:
            break
        u, v = e
        a, b = values[u], values[v]
        s, p = a + b, a * b
        # Prefer edges that add few new distinct sums/products
        new_sums = 0 if s in sums else 1
        new_prods = 0 if p in prods else 1
        cost = new_sums + new_prods
        edges.append((e, cost))
    
    # Sort by cost and take best m
    edges.sort(key=lambda x: x[1])
    edges = [e for e, _ in edges[:m]]
    
    # Local improvement
    edges_set = set(edges)
    best_score = _score(values, list(edges_set))
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if best_score <= 2:
            break
        # Try swapping
        u, v = random.choice(list(edges_set))
        candidates = [e for e in all_edges if e not in edges_set]
        if not candidates:
            break
        x, y = random.choice(candidates)
        new_edges = set(edges_set)
        new_edges.discard((u, v))
        new_edges.add((x, y))
        new_score = _score(values, list(new_edges))
        if new_score <= best_score:
            edges_set = new_edges
            best_score = new_score
    
    return best_score, [list(e) for e in edges_set]

def _hill_climb_edges(values, m, n, max_iter=30000):
    """Hill climbing with incremental scoring."""
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    edges = set(random.sample(all_edges, m))
    
    # Precompute sums and products
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        u, v = e
        edge_sums[e] = values[u] + values[v]
        edge_prods[e] = values[u] * values[v]
    
    current_sums = set(edge_sums[e] for e in edges)
    current_prods = set(edge_prods[e] for e in edges)
    current_score = max(len(current_sums), len(current_prods))
    best_score = current_score
    best_edges = set(edges)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 2:
            break
        u, v = random.choice(list(edges))
        candidates = [e for e in all_edges if e not in edges]
        if not candidates:
            break
        x, y = random.choice(candidates)
        
        # Compute new score incrementally
        old_s, old_p = edge_sums[(u, v)], edge_prods[(u, v)]
        new_s, new_p = edge_sums[(x, y)], edge_prods[(x, y)]
        
        # Count how many distinct sums/products we'd lose
        sum_count = {s: 0 for s in current_sums}
        prod_count = {p: 0 for p in current_prods}
        for e in edges:
            sum_count[edge_sums[e]] += 1
            prod_count[edge_prods[e]] += 1
        
        new_sum_count = dict(sum_count)
        new_prod_count = dict(prod_count)
        new_sum_count[old_s] -= 1
        new_prod_count[old_p] -= 1
        if new_sum_count[old_s] == 0:
            del new_sum_count[old_s]
        if new_prod_count[old_p] == 0:
            del new_prod_count[old_p]
        
        if new_s in new_sum_count:
            new_sum_count[new_s] += 1
        else:
            new_sum_count[new_s] = 1
        if new_p in new_prod_count:
            new_prod_count[new_p] += 1
        else:
            new_prod_count[new_p] = 1
        
        new_score = max(len(new_sum_count), len(new_prod_count))
        if new_score <= current_score:
            edges.discard((u, v))
            edges.add((x, y))
            current_sums = set(new_sum_count.keys())
            current_prods = set(new_prod_count.keys())
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
    
    return best_score, [list(e) for e in best_edges]

def _try_value_set(values, m, n):
    """Try a value set with both greedy and hill climbing."""
    if time.time() > _DEADLINE:
        return None
    
    # Try greedy
    greedy_score, greedy_edges = _greedy_edges(values, m, n, max_iter=2000)
    
    # Try hill climbing from greedy result
    best_score = greedy_score
    best_edges = greedy_edges
    
    if time.time() < _DEADLINE - 5:
        hc_score, hc_edges = _hill_climb_edges(values, m, n, max_iter=10000)
        if hc_score < best_score:
            best_score = hc_score
            best_edges = hc_edges
    
    return best_score, values, best_edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case: all distinct
    best_result = None
    
    # Candidate value sets to try
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 4. Powers of 2 (known to give unique sums/products, but try anyway)
    candidates.append([2 ** i for i in range(n)])
    
    # 5. Mixed small values with lots of collisions
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 6. Try some random small value sets
    random.seed(42)
    for _ in range(20):
        vals = sorted(random.sample(range(1, 50), n))
        candidates.append(vals)
    
    # 7. Arithmetic progressions
    for start in range(1, 6):
        for step in range(1, 4):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                candidates.append(vals)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        # Ensure all are Python int
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
