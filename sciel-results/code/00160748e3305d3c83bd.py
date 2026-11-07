"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

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

def _score_detail(values, edges):
    """Return (num_distinct_sums, num_distinct_products, sum_set, prod_set)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods), sums, prods

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
    
    # Initialize with a structured approach
    initial_values = list(range(1, n + 1))
    initial_edges = [list(e) for e in all_pairs[:m]]
    best_score = _score(initial_values, initial_edges)
    best_values = initial_values.copy()
    best_edges = initial_edges.copy()
    
    def _valid_values(vals):
        if len(vals) != n:
            return False
        if len(set(vals)) != n:
            return False
        if not all(isinstance(v, int) and 1 <= v <= max_val for v in vals):
            return False
        return True
    
    def _valid_edges(edges):
        if len(edges) != m:
            return False
        seen = set()
        for u, v in edges:
            if not (isinstance(u, int) and isinstance(v, int)):
                return False
            if not (0 <= u < n and 0 <= v < n):
                return False
            if u == v:
                return False
            t = (min(u, v), max(u, v))
            if t in seen:
                return False
            seen.add(t)
        return True
    
    def gen_geometric_values():
        """Generate geometric progression values for product structure."""
        ratios = [2, 3, 4]
        base = random.choice([1, 2, 3, 5, 7])
        ratio = random.choice(ratios)
        vals = []
        for i in range(n):
            v = base * (ratio ** i)
            if v <= max_val:
                vals.append(v)
            else:
                vals.append(base * (ratio ** (i - 1)) * random.randint(1, ratio))
        # Ensure distinctness
        seen = set()
        unique_vals = []
        for v in vals:
            if v not in seen and 1 <= v <= max_val:
                seen.add(v)
                unique_vals.append(v)
        while len(unique_vals) < n:
            v = random.randint(1, max_val)
            if v not in seen:
                seen.add(v)
                unique_vals.append(v)
        return unique_vals
    
    def gen_multiplicative_values():
        """Generate values with rich multiplicative structure."""
        small_factors = [2, 3, 5, 7, 11, 13]
        vals = set()
        attempts = 0
        while len(vals) < n and attempts < 1000:
            # Pick 2-4 small prime factors
            k = random.randint(2, 4)
            factors = [random.choice(small_factors) for _ in range(k)]
            v = 1
            for f in factors:
                v *= f
            if 1 <= v <= max_val:
                vals.add(v)
            attempts += 1
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def gen_colliding_sum_values():
        """Generate values that create many sum collisions."""
        # Use pairs that sum to the same value
        target_sum = random.randint(100, 500)
        vals = set()
        for i in range(n // 2):
            a = random.randint(1, target_sum - 1)
            b = target_sum - a
            if a != b and a > 0 and b > 0 and a <= max_val and b <= max_val:
                vals.add(a)
                vals.add(b)
        # Fill remaining
        while len(vals) < n:
            v = random.randint(1, max_val)
            if v not in vals:
                vals.add(v)
        return list(vals)
    
    def gen_structured_values(strategy):
        """Generate values with specific structural properties."""
        if strategy == 'arithmetic':
            step = random.choice([1, 2, 3, 4, 5])
            start = random.randint(1, max_val - step * (n - 1))
            return [start + i * step for i in range(n)]
        elif strategy == 'geometric':
            return gen_geometric_values()
        elif strategy == 'multiplicative':
            return gen_multiplicative_values()
        elif strategy == 'sum_collision':
            return gen_colliding_sum_values()
        else:
            return random.sample(range(1, max_val + 1), n)
    
    def greedy_edge_selection(values, target_edges=m):
        """Select edges greedily to maximize collision potential."""
        edge_set = set()
        edges = []
        
        # Precompute all possible sums and products
        sum_counts = {}
        prod_counts = {}
        for i, j in all_pairs:
            a, b = values[i], values[j]
            s = a + b
            p = a * b
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        # Score each edge by how much it contributes to collisions
        scored_edges = []
        for i, j in all_pairs:
            a, b = values[i], values[j]
            s = a + b
            p = a * b
            # Edge is good if its sum/product is shared with other potential edges
            score = sum_counts[s] + prod_counts[p]
            scored_edges.append((score, i, j))
        
        scored_edges.sort(key=lambda x: -x[0])
        
        for score, i, j in scored_edges:
            if len(edges) >= target_edges:
                break
            t = (min(i, j), max(i, j))
            if t not in edge_set:
                edge_set.add(t)
                edges.append([i, j])
        
        # Fill if needed
        if len(edges) < target_edges:
            remaining = [e for e in all_pairs if (min(e[0], e[1]), max(e[0], e[1])) not in edge_set]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= target_edges:
                    break
                edges.append(list(e))
                edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        return edges
    
    def _quick_score(values, edges):
        """Fast score computation."""
        sums = set()
        prods = set()
        for u, v in edges:
            a, b = values[u], values[v]
            sums.add(a + b)
            prods.add(a * b)
        return max(len(sums), len(prods))
    
    def hill_climb_values(values, edges, max_rounds=30):
        """Hill climbing on values with targeted mutations."""
        current_score = _quick_score(values, edges)
        values_set = set(values)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                old_val = values[i]
                values_set.discard(old_val)
                
                # Generate targeted candidates
                candidates = set()
                
                # Small perturbations of existing values
                for v in values:
                    for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                        nv = v + delta
                        if 1 <= nv <= max_val:
                            candidates.add(nv)
                
                # Multiplicative relationships
                for factor in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                    candidates.add(factor * random.randint(1, max_val // factor))
                
                # Sum-targeting: try to match sums of other edges
                for u, v in edges:
                    if u != i and v != i:
                        s = values[u] + values[v]
                        # Try values that would create same sum with some other vertex
                        for j in range(n):
                            if j != i:
                                nv = s - values[j]
                                if 1 <= nv <= max_val:
                                    candidates.add(nv)
                
                # Random candidates
                for _ in range(20):
                    candidates.add(random.randint(1, max_val))
                
                best_new_val = old_val
                best_new_score = current_score
                
                for new_val in candidates:
                    if new_val == old_val or new_val in values_set:
                        continue
                    values[i] = new_val
                    values_set.add(new_val)
                    new_score = _quick_score(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    values[i] = old_val
                    values_set.discard(new_val)
                
                if best_new_val != old_val:
                    values[i] = best_new_val
                    values_set.discard(old_val)
                    values_set.add(best_new_val)
                    current_score = best_new_score
                    improved = True
                else:
                    values_set.add(old_val)
            
            if not improved:
                break
        
        return values, current_score
    
    def hill_climb_edges(values, edges, max_rounds=20):
        """Hill climbing on edge selection."""
        current_score = _quick_score(values, edges)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for idx in range(m):
                e_remove = tuple(sorted(edges[idx]))
                
                # Find best replacement
                candidate_edges = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]
                
                # Score candidates by collision potential
                scored = []
                for e in candidate_edges:
                    u, v = e
                    a, b = values[u], values[v]
                    s = a + b
                    p = a * b
                    # Count how many current edges share this sum/product
                    sum_match = sum(1 for eu, ev in edges if eu != u and ev != v and values[eu] + values[ev] == s)
                    prod_match = sum(1 for eu, ev in edges if eu != u and ev != v and values[eu] * values[ev] == p)
                    scored.append((sum_match + prod_match, e))
                
                scored.sort(key=lambda x: -x[0])
                top_candidates = [e for _, e in scored[:15]]
                random.shuffle(top_candidates)
                
                best_e_add = None
                best_new_score = current_score
                for e_add in top_candidates:
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _quick_score(values, new_edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_e_add = e_add
                
                if best_e_add is not None:
                    edges = edges[:idx] + edges[idx+1:] + [list(best_e_add)]
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(best_e_add)))
                    current_score = best_new_score
                    improved = True
            
            if not improved:
                break
        
        return edges, current_score
    
    def tabu_search(values, edges, max_iter=200, tabu_len=10):
        """Tabu search for joint optimization."""
        current_score = _quick_score(values, edges)
        best_score_ts = current_score
        best_values_ts = values.copy()
        best_edges_ts = edges.copy()
        
        tabu_values = []  # (vertex_index, value, expiry)
        tabu_edges = []   # (edge_tuple, expiry)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Expire old tabu entries
            tabu_values = [(v, val, exp) for v, val, exp in tabu_values if exp > iteration]
            tabu_edges = [(e, exp) for e, exp in tabu_edges if exp > iteration]
            
            # Generate a random move
            if random.random() < 0.5:
                # Change a value
                i = random.randint(0, n - 1)
                old_val = values[i]
                new_val = random.randint(1, max_val)
                # Avoid tabu
                if any(v == i and val == new_val for v, val, exp in tabu_values):
                    new_val = random.randint(1, max_val)
                
                values[i] = new_val
                tabu_values.append((i, new_val, iteration + tabu_len))
                tabu_values.append((i, old_val, iteration + tabu_len))
                
                new_score = _quick_score(values, edges)
            else:
                # Change an edge
                idx = random.randint(0, m - 1)
                old_edge = tuple(sorted(edges[idx]))
                available = [e for e in all_pairs if tuple(sorted(e)) not in set(tuple(sorted(e2)) for e2 in edges)]
                if not available:
                    continue
                new_edge = random.choice(available)
                
                if any(e == tuple(sorted(new_edge)) for e, exp in tabu_edges):
                    continue
                
                edges[idx] = list(new_edge)
                tabu_edges.append((tuple(sorted(new_edge)), iteration + tabu_len))
                tabu_edges.append((old_edge, iteration + tabu_len))
                
                new_score = _quick_score(values, edges)
            
            # Accept if better or random (with cooling)
            delta = new_score - current_score
            temperature = max(0.1, 3.0 * (1 - iteration / max_iter))
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_score = new_score
                
                if current_score < best_score_ts:
                    best_score_ts = current_score
                    best_values_ts = values.copy()
                    best_edges_ts = edges.copy()
            else:
                # Revert
                if random.random() < 0.5:
                    # Revert value
                    i = random.randint(0, n - 1)
                    # Find what to revert to - this is complex, just keep going
                    pass
                else:
                    # Revert edge
                    idx = random.randint(0, m - 1)
                    pass
        
        return best_values_ts, best_edges_ts, best_score_ts
    
    # Main search loop with diverse strategies
    strategies = ['arithmetic', 'geometric', 'multiplicative', 'sum_collision', 'random']
    
    for iteration in range(60):
        if time.time() > deadline:
            break
        
        # Choose strategy based on iteration
        if iteration < 10:
            strategy = strategies[iteration % len(strategies)]
            cur_values = gen_structured_values(strategy)
            cur_edges = greedy_edge_selection(cur_values)
        elif iteration < 30:
            # Local search from best
            cur_values = best_values.copy()
            cur_edges = best_edges.copy()
        else:
            # Mix: sometimes restart, sometimes continue
            if random.random() < 0.3:
                strategy = random.choice(strategies)
                cur_values = gen_structured_values(strategy)
                cur_edges = greedy_edge_selection(cur_values)
            else:
                cur_values = best_values.copy()
                cur_edges = best_edges.copy()
        
        # Optimize
        if time.time() < deadline - 5:
            cur_values, vscore = hill_climb_values(cur_values, cur_edges, max_rounds=20)
        
        if time.time() < deadline - 5:
            cur_edges, escore = hill_climb_edges(cur_values, cur_edges, max_rounds=15)
        
        cur_score = _quick_score(cur_values, cur_edges)
        if cur_score < best_score:
            best_score = cur_score
            best_values = cur_values.copy()
            best_edges = cur_edges.copy()
        
        # Occasional tabu search
        if iteration % 8 == 4 and time.time() < deadline - 15:
            ts_values, ts_edges, ts_score = tabu_search(best_values.copy(), best_edges.copy(), max_iter=100)
            if ts_score < best_score:
                best_score = ts_score
                best_values = ts_values.copy()
                best_edges = ts_edges.copy()
        
        if best_score <= 15:
            break
    
    # Final polish
    if time.time() < deadline - 10:
        best_values, _ = hill_climb_values(best_values, best_edges, max_rounds=15)
        best_edges, _ = hill_climb_edges(best_values, best_edges, max_rounds=10)
        final_score = _quick_score(best_values, best_edges)
        if final_score < best_score:
            best_score = final_score
    
    # Final validation
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
