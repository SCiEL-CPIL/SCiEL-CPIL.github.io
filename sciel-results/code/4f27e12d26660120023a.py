"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    all_pairs = list(combinations(range(n), 2))
    n_pairs = len(all_pairs)  # 91
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Key insight: use values that are multiples of a common factor to create product collisions
    # If all values are multiples of k, then all products are multiples of k^2
    # For sums: if values form an arithmetic progression, sums cluster
    
    def gen_structured_values(base, step, start):
        """Generate values as base + i*step for structured sum collisions."""
        vals = [start + i * step for i in range(n)]
        if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
            return vals
        return None
    
    def gen_common_multiple_values(k):
        """Generate n distinct multiples of k up to max_val."""
        candidates = [k * i for i in range(1, max_val // k + 1) if k * i <= max_val]
        if len(candidates) >= n:
            return random.sample(candidates, n)
        return None
    
    def gen_geometric_values(ratio, start):
        """Generate values as geometric progression for product structure."""
        vals = []
        v = start
        for i in range(n):
            vals.append(v)
            v = int(v * ratio)
            if v > max_val:
                return None
        if all(1 <= x <= max_val for x in vals) and len(set(vals)) == n:
            return vals
        return None
    
    def greedy_edge_selection(values, n_edges=m):
        """Select edges to maximize sum/product collisions using collision graph."""
        # Precompute all pair sums and products
        pair_info = {}
        for i, j in all_pairs:
            a, b = values[i], values[j]
            pair_info[(i, j)] = (a + b, a * b)
        
        # Group edges by sum and by product
        sum_groups = defaultdict(list)
        prod_groups = defaultdict(list)
        for (i, j), (s, p) in pair_info.items():
            sum_groups[s].append((i, j))
            prod_groups[p].append((i, j))
        
        # Score each edge by how many collisions it creates
        edge_scores = {}
        for (i, j) in all_pairs:
            s, p = pair_info[(i, j)]
            score = len(sum_groups[s]) + len(prod_groups[p])
            edge_scores[(i, j)] = score
        
        # Greedy: pick highest scoring edges, updating scores as we go
        selected = []
        selected_set = set()
        remaining_sums = defaultdict(int)
        remaining_prods = defaultdict(int)
        
        for (i, j), (s, p) in pair_info.items():
            remaining_sums[s] += 1
            remaining_prods[p] += 1
        
        for _ in range(n_edges):
            best_edge = None
            best_score = -1
            for (i, j) in all_pairs:
                if (i, j) in selected_set:
                    continue
                s, p = pair_info[(i, j)]
                # Score: how many other selected edges share this sum or product
                # Plus potential future collisions
                local_score = remaining_sums[s] + remaining_prods[p]
                if local_score > best_score:
                    best_score = local_score
                    best_edge = (i, j)
            
            if best_edge is None:
                break
            
            i, j = best_edge
            s, p = pair_info[(i, j)]
            selected.append([i, j])
            selected_set.add((i, j))
            remaining_sums[s] -= 1
            remaining_prods[p] -= 1
        
        return selected
    
    def local_search(values, edges, max_iter=100):
        """Local search: try swapping edges and adjusting values."""
        current_score = _score(values, edges)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try edge swaps
            for idx in range(m):
                if time.time() > deadline:
                    break
                e_remove = tuple(sorted(edges[idx]))
                candidates = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]
                
                best_add = None
                best_new_score = current_score
                for e_add in random.sample(candidates, min(30, len(candidates))):
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _score(values, new_edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_add = e_add
                
                if best_add is not None:
                    edges = edges[:idx] + edges[idx+1:] + [list(best_add)]
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(best_add)))
                    current_score = best_new_score
                    improved = True
            
            # Try value mutations
            for _ in range(5):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                old_val = values[i]
                new_val = random.randint(1, max_val)
                while new_val == old_val or new_val in set(values[:i] + values[i+1:]):
                    new_val = random.randint(1, max_val)
                
                values[i] = new_val
                new_score = _score(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    values[i] = old_val
            
            if not improved:
                break
        
        return values, edges, current_score
    
    # Try multiple structured value generation strategies
    strategies = []
    
    # Strategy 1: Common multiples
    for k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        vals = gen_common_multiple_values(k)
        if vals:
            strategies.append(vals)
    
    # Strategy 2: Arithmetic progressions
    for step in [1, 2, 3, 5, 10, 20, 50, 100]:
        for start in [1, 2, 5, 10, 50, 100]:
            vals = gen_structured_values(1, step, start)
            if vals:
                strategies.append(vals)
    
    # Strategy 3: Geometric-like
    for ratio in [2, 3]:
        for start in [1, 2, 3]:
            vals = gen_geometric_values(ratio, start)
            if vals:
                strategies.append(vals)
    
    # Strategy 4: Random
    for _ in range(10):
        strategies.append(random.sample(range(1, max_val + 1), n))
    
    # Evaluate each strategy
    for vals in strategies:
        if time.time() > deadline:
            break
        
        edges = greedy_edge_selection(vals)
        score = _score(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals.copy()
            best_edges = edges.copy()
        
        # Local search from this starting point
        if time.time() < deadline - 10:
            v, e, s = local_search(vals.copy(), [list(x) for x in edges], max_iter=50)
            if s < best_score:
                best_score = s
                best_values = v
                best_edges = e
        
        if best_score <= 14:
            break
    
    # Final polish
    if time.time() < deadline - 5:
        v, e, s = local_search(best_values.copy(), [list(x) for x in best_edges], max_iter=50)
        if s < best_score:
            best_score = s
            best_values = v
            best_edges = e
    
    # Validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
