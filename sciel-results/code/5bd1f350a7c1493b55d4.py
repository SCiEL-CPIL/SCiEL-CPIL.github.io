"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _compute_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_edges(n, m):
    """Return a specific set of m edges out of all possible n*(n-1)/2 edges."""
    # We'll use a greedy approach later, but for now, let's just pick the first m
    # Actually, we need to optimize edge selection too.
    # Let's return all possible edges first, and then select in construct.
    return list(combinations(range(n), 2))

def _select_edges(values, n, m, deadline):
    """Select m edges to minimize max(distinct sums, distinct products)."""
    all_edges = list(combinations(range(n), 2))
    # We want to choose m edges such that the number of distinct sums and products is minimized.
    # A greedy approach: at each step, add the edge that minimizes the increase in distinct sums/products.
    
    # Precompute sum and product for each edge
    edge_info = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_info[(i, j)] = (s, p)
    
    selected = []
    current_sums = set()
    current_prods = set()
    
    remaining = list(all_edges)
    random.shuffle(remaining)
    
    while len(selected) < m and time.time() < deadline:
        best_edge = None
        best_score = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        # Try each remaining edge
        for e in remaining:
            s, p = edge_info[e]
            new_sums = current_sums | {s}
            new_prods = current_prods | {p}
            score = max(len(new_sums), len(new_prods))
            if score < best_score:
                best_score = score
                best_edge = e
                best_new_sums = new_sums
                best_new_prods = new_prods
        
        if best_edge is None:
            break
        
        selected.append(best_edge)
        current_sums = best_new_sums
        current_prods = best_new_prods
        remaining.remove(best_edge)
    
    return [list(e) for e in selected]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10 second safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    best_values = list(range(1, n + 1))
    best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    best_score = _compute_score(best_values, best_edges)
    
    # Try multiple random value sets and optimize edges for each
    random.seed(42)
    
    # Generate candidate value sets
    candidate_sets = []
    
    # 1. Sequential
    candidate_sets.append(list(range(1, n + 1)))
    
    # 2. Random sets
    for _ in range(200):
        if time.time() > deadline:
            break
        vals = random.sample(range(1, max_val + 1), n)
        candidate_sets.append(vals)
    
    # 3. Sets with small values to encourage collisions
    for _ in range(100):
        if time.time() > deadline:
            break
        vals = random.sample(range(1, 50), n)
        candidate_sets.append(vals)
    
    # 4. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 20):
            if time.time() > deadline:
                break
            vals = [start + i * step for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                candidate_sets.append(vals)
    
    # Evaluate each candidate
    for vals in candidate_sets:
        if time.time() > deadline:
            break
        
        # Ensure values are distinct positive integers <= 10000
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > max_val for v in vals):
            continue
        
        # Select optimal edges for this value set
        edges = _select_edges(vals, n, m, deadline)
        
        if len(edges) < m:
            continue
        
        score = _compute_score(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges
    
    # Local search on the best values found
    if time.time() < deadline - 5:
        current_values = best_values[:]
        current_edges = best_edges[:]
        current_score = best_score
        
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            # Try to improve by changing one value
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                
                old_val = current_values[idx]
                
                # Try nearby values
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in current_values:
                        continue
                    
                    test_values = current_values[:]
                    test_values[idx] = new_val
                    
                    # Re-select edges
                    test_edges = _select_edges(test_values, n, m, deadline)
                    if len(test_edges) < m:
                        continue
                    
                    test_score = _compute_score(test_values, test_edges)
                    
                    if test_score < current_score:
                        current_values = test_values
                        current_edges = test_edges
                        current_score = test_score
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values
            best_edges = current_edges
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) for v in best_values)
    assert all(1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    for e in best_edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n
        assert 0 <= e[1] < n
    
    # Check for duplicate edges
    edge_set = set()
    for e in best_edges:
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
