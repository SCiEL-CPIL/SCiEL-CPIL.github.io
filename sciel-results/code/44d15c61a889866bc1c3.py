"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edges(values, all_edges, rng):
        """Greedy: pick edges that minimize the increase in distinct sums/products."""
        sums_seen = set()
        prods_seen = set()
        edges = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_s, best_p = 0, 0
            for e in remaining:
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]
                # Score: 0 if already seen, 1 if new. We want to minimize new ones.
                # Tie-breaker: prefer edges with smaller values? Or just random.
                score = (0 if s in sums_seen else 1) + (0 if p in prods_seen else 1)
                if score < best_score:
                    best_score = score
                    best_edge = e
                    best_s, best_p = s, p
                    if best_score == 0:
                        break
            if best_edge is None:
                best_edge = remaining[0]
                best_s = values[best_edge[0]] + values[best_edge[1]]
                best_p = values[best_edge[0]] * values[best_edge[1]]
            edges.append(best_edge)
            remaining.remove(best_edge)
            sums_seen.add(best_s)
            prods_seen.add(best_p)
        
        return edges

    def local_search(values, edges, rng, time_limit=2.0):
        """Local search to improve edge selection."""
        current_edges = [list(e) for e in edges]
        current_score = evaluate(values, current_edges)
        
        best_edges = [list(e) for e in current_edges]
        best_score = current_score
        
        all_edges = list(combinations(range(n), 2))
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        
        start_t = time.time()
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start_t < time_limit and no_improve < max_no_improve:
            if time.time() > deadline - 5:
                break
            
            # Try swapping an edge
            idx = rng.randrange(len(current_edges))
            old_edge = tuple(sorted(current_edges[idx]))
            
            # Find a non-edge
            non_edges = [e for e in all_edges if e not in edge_set]
            if not non_edges:
                break
                
            new_edge = rng.choice(non_edges)
            
            # Apply move
            new_edges = [list(e) for e in current_edges]
            new_edges[idx] = list(new_edge)
            new_edge_set = set(tuple(sorted(e)) for e in new_edges)
            
            new_score = evaluate(values, new_edges)
            
            if new_score <= current_score:
                current_edges = new_edges
                edge_set = new_edge_set
                current_score = new_score
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [list(e) for e in current_edges]
            else:
                no_improve += 1
        
        return best_edges, best_score

    def value_local_search(values, edges, rng, time_limit=2.0):
        """Local search to improve values."""
        current_values = list(values)
        current_score = evaluate(current_values, edges)
        
        best_values = list(current_values)
        best_score = current_score
        
        start_t = time.time()
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start_t < time_limit and no_improve < max_no_improve:
            if time.time() > deadline - 5:
                break
            
            # Try changing one value
            idx = rng.randrange(n)
            old_val = current_values[idx]
            
            # Try a nearby value
            if rng.random() < 0.5:
                new_val = old_val + rng.randint(-5, 5)
            else:
                new_val = rng.randint(1, max_val)
            
            if new_val < 1 or new_val > max_val:
                continue
            
            # Check distinctness
            new_values = list(current_values)
            new_values[idx] = new_val
            if len(set(new_values)) != n:
                continue
            
            new_score = evaluate(new_values, edges)
            
            if new_score <= current_score:
                current_values = new_values
                current_score = new_score
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
            else:
                no_improve += 1
        
        return best_values, best_score

    def simulated_annealing(values, edges, rng, time_limit=2.0):
        """Improve solution using simulated annealing."""
        current_values = list(values)
        current_edges = [list(e) for e in edges]
        current_score = evaluate(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = [list(e) for e in current_edges]
        best_score = current_score
        
        all_edges = list(combinations(range(n), 2))
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.99
        
        start_t = time.time()
        while time.time() - start_t < time_limit and T > T_min:
            if time.time() > deadline - 5:
                break
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # Swap an edge
                idx = rng.randrange(len(current_edges))
                old_edge = tuple(sorted(current_edges[idx]))
                
                # Find a non-edge
                non_edges = [e for e in all_edges if e not in edge_set]
                if not non_edges:
                    T *= cooling_rate
                    continue
                
                new_edge = rng.choice(non_edges)
                
                # Apply move
                new_edges = [list(e) for e in current_edges]
                new_edges[idx] = list(new_edge)
                new_edge_set = set(tuple(sorted(e)) for e in new_edges)
                
                new_score = evaluate(current_values, new_edges)
                
                # Acceptance
                if new_score < current_score or rng.random() < math.exp(-(new_score - current_score) / T):
                    current_edges = new_edges
                    edge_set = new_edge_set
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = [list(e) for e in current_edges]
                        best_values = list(current_values)
            else:
                # Swap two values
                i, j = rng.sample(range(n), 2)
                new_values = list(current_values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                
                if len(set(new_values)) != n:
                    T *= cooling_rate
                    continue
                
                new_score = evaluate(new_values, current_edges)
                
                # Acceptance
                if new_score < current_score or rng.random() < math.exp(-(new_score - current_score) / T):
                    current_values = new_values
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = [list(e) for e in current_edges]
            
            T *= cooling_rate
        
        return best_values, best_edges, best_score

    rng = random.Random(42)
    all_edges = list(combinations(range(n), 2))
    
    best_overall = float('inf')
    best_result = None
    
    # Generate diverse value sets
    value_sets = []
    
    # 1. Small consecutive integers
    for start in range(1, 30):
        vals = list(range(start, start + n))
        if max(vals) <= max_val:
            value_sets.append(vals)
            
    # 2. Arithmetic progressions
    for start in range(1, 50):
        for step in range(1, 10):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
                
    # 3. Random small values (biased towards small numbers to encourage collisions)
    for _ in range(200):
        # Sample from 1..50, ensuring distinct
        vals = rng.sample(range(1, 50), n)
        value_sets.append(vals)
        
    # 4. Multiples of small numbers (encourages product collisions)
    for base in [2, 3, 4, 5, 6]:
        for offset in range(0, 5):
            vals = [base * (i + 1) + offset for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                value_sets.append(vals)
                
    # 5. Powers (less likely to help, but check)
    for base in [2, 3]:
        vals = [base ** i for i in range(n)]
        if max(vals) <= max_val:
            value_sets.append(vals)
            
    # 6. Values with common factor structure
    for g in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for _ in range(20):
            # Generate n distinct multiples of g
            bases = rng.sample(range(1, max_val // g + 1), n)
            vals = [b * g for b in bases]
            if len(set(vals)) == n and max(vals) <= max_val:
                value_sets.append(vals)
                
    # 7. Mix of small and large values
    for _ in range(100):
        half_small = n // 2
        small_vals = rng.sample(range(1, 30), half_small)
        large_vals = rng.sample(range(50, 200), n - half_small)
        vals = small_vals + large_vals
        if len(set(vals)) == n:
            value_sets.append(vals)

    # Deduplicate
    seen_sets = set()
    unique_value_sets = []
    for vs in value_sets:
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
            
    # Shuffle value sets to try diverse ones first
    rng.shuffle(unique_value_sets)
    
    # Run search for each value set
    for vs in unique_value_sets:
        if time.time() > deadline - 10:
            break
            
        # Greedy edge initialization
        edges = greedy_edges(vs, all_edges, rng)
        
        # Local search on edges
        if time.time() > deadline - 15:
            break
        edges2, score2 = local_search(vs, edges, rng, time_limit=min(2.0, max(0.5, deadline - time.time() - 15)))
        if score2 < best_overall:
            best_overall = score2
            best_result = (list(vs), [list(e) for e in edges2])
        
        # Value local search
        if time.time() > deadline - 15:
            break
        vs2, score3 = value_local_search(vs, edges2, rng, time_limit=min(2.0, max(0.5, deadline - time.time() - 15)))
        if score3 < best_overall:
            best_overall = score3
            best_result = (list(vs2), [list(e) for e in edges2])
            # Re-optimize edges with new values
            edges3 = greedy_edge_select(vs2)
            score4 = evaluate(vs2, edges3)
            if score4 > best_overall:
                edges3, score4 = local_search(vs2, edges3, rng, time_limit=min(1.0, max(0.3, deadline - time.time() - 15)))
            if score4 < best_overall:
                best_overall = score4
                best_result = (list(vs2), [list(e) for e in edges3])
    
    if best_result is None:
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
