"""Rewritten program combining MODEL-AND-OPTIMIZE with structural value-pair collision search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val_limit = 10000
    
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        """Calculate the score: max(distinct sums, distinct products)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_evaluate(values, edges):
        """Faster evaluation using local sets."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def optimize_edges(values, edges, rng, time_limit):
        """
        Optimize the set of edges for a fixed set of values.
        Uses local search: swap one edge with another to reduce score.
        """
        current_score = evaluate(values, edges)
        edges_set = set(tuple(e) for e in edges)
        
        improved = True
        no_improve_counter = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve_counter = 0
            # Try random swaps
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    return edges, current_score
                
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                new_edge_tuple = tuple(new_edge)
                if new_edge_tuple in edges_set:
                    continue
                
                old_edge = edges[i]
                old_edge_tuple = tuple(old_edge)
                
                # Apply swap
                edges[i] = new_edge
                edges_set.discard(old_edge_tuple)
                edges_set.add(new_edge_tuple)
                
                new_score = evaluate(values, edges)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improve_counter = 0
                else:
                    # Revert
                    edges[i] = old_edge
                    edges_set.discard(new_edge_tuple)
                    edges_set.add(old_edge_tuple)
                    no_improve_counter += 1
                    if no_improve_counter > 800:
                        break
        
        return edges, current_score

    def optimize_values(values, edges, rng, time_limit):
        """
        Optimize the set of values for a fixed set of edges.
        Uses local search: change one value to another small integer.
        """
        current_score = evaluate(values, edges)
        values_set = set(values)
        
        improved = True
        no_improve_counter = 0
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            no_improve_counter = 0
            for _ in range(150):
                if (time.time() - start_time) > time_limit:
                    return values, current_score
                
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                
                # Generate candidates
                candidates = []
                # Nearby values
                for d in range(-8, 9):
                    candidates.append(old_val + d)
                # Random small values
                for _ in range(15):
                    candidates.append(rng.randint(1, 60))
                
                for new_val in candidates:
                    if new_val < 1 or new_val > max_val_limit:
                        continue
                    if new_val in values_set:
                        continue
                    
                    values[idx] = new_val
                    values_set.discard(old_val)
                    values_set.add(new_val)
                    
                    new_score = evaluate(values, edges)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_counter = 0
                        break # Accept and move to next iteration
                    else:
                        # Revert
                        values[idx] = old_val
                        values_set.discard(new_val)
                        values_set.add(old_val)
                        no_improve_counter += 1
                        if no_improve_counter > 1500:
                            break
            if no_improve_counter > 1500:
                break
        
        return values, current_score

    def find_structural_values(rng, time_limit):
        """
        Structural approach: find value sets that naturally create many collisions.
        We look for sets where small value pairs share sums/products.
        """
        best_score = float('inf')
        best_values = None
        
        # Generate candidate value sets with structural properties
        candidates = []
        
        # Consecutive integers
        candidates.append(list(range(1, n + 1)))
        # Even integers
        candidates.append(list(range(2, 2 * n + 1, 2)))
        # Multiples of 3
        candidates.append(list(range(3, 3 * n + 1, 3)))
        # Small random sets
        for _ in range(30):
            if time.time() > time_limit:
                break
            vals = sorted(rng.sample(range(1, 40), n))
            candidates.append(vals)
        
        # Structured: pairs that share sums
        # E.g., [a, b, c, d] where a+d = b+c
        for base in range(1, 10):
            vals = []
            for i in range(n):
                # Create symmetric pairs
                if i < n // 2:
                    vals.append(base + i)
                else:
                    vals.append(base + (n - 1 - i))
            if len(set(vals)) == n:
                candidates.append(sorted(vals))
        
        # Evaluate each candidate
        for vals in candidates:
            if len(set(vals)) != n or any(v < 1 or v > max_val_limit for v in vals):
                continue
            
            # Use all edges for evaluation (worst case)
            sums = set()
            prods = set()
            for u, v in all_edges:
                sums.add(vals[u] + vals[v])
                prods.add(vals[u] * vals[v])
            
            score = max(len(sums), len(prods))
            
            if score < best_score:
                best_score = score
                best_values = list(vals)
        
        # Local search to improve
        if best_values is not None and time.time() < time_limit - 15:
            current = list(best_values)
            current_score = best_score
            values_set = set(current)
            
            for iteration in range(50):
                if time.time() > time_limit - 5:
                    break
                
                improved = False
                for idx in range(n):
                    old_val = current[idx]
                    for delta in range(-12, 13):
                        new_val = old_val + delta
                        if new_val < 1 or new_val > max_val_limit:
                            continue
                        if new_val in values_set:
                            continue
                        
                        current[idx] = new_val
                        values_set.discard(old_val)
                        values_set.add(new_val)
                        
                        # Evaluate with all edges
                        sums = set()
                        prods = set()
                        for u, v in all_edges:
                            sums.add(current[u] + current[v])
                            prods.add(current[u] * current[v])
                        
                        new_score = max(len(sums), len(prods))
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            current[idx] = old_val
                            values_set.discard(new_val)
                            values_set.add(old_val)
                
                if not improved:
                    break
                
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current)
        
        return best_values, best_score

    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Phase 1: Find good structural values
    structural_values, structural_score = find_structural_values(rng, deadline - 30.0)
    
    if structural_values is not None:
        # Optimize edges for these values
        edges = rng.sample(all_edges, m)
        edges, score = optimize_edges(structural_values, edges, rng, deadline - 20.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(structural_values), 'edges': [list(e) for e in edges]}
        
        # Optimize values further
        values = list(structural_values)
        values, score2 = optimize_values(values, edges, rng, deadline - 10.0)
        
        if score2 < best_overall_score:
            best_overall_score = score2
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Phase 2: Try different initial strategies
    initial_strategies = [
        list(range(1, n + 1)),
        list(range(2, 2 * n + 1, 2)),
        list(range(3, 3 * n + 1, 3)),
        sorted(rng.sample(range(1, 50), n)),
        sorted(rng.sample(range(1, 100), n)),
    ]
    
    for init_values in initial_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
            
        # Ensure distinctness
        if len(set(init_values)) != n:
            continue
            
        # 1. Initial edges: Random sample of 35 edges
        edges = rng.sample(all_edges, m)
        
        # 2. Optimize edges
        edges, score = optimize_edges(init_values, edges, rng, deadline - 15.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': [list(e) for e in edges]}
        
        # 3. Optimize values
        values = list(init_values)
        values, score2 = optimize_values(values, edges, rng, deadline - 10.0)
        
        if score2 < best_overall_score:
            best_overall_score = score2
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
            
        # 4. Re-optimize edges with new values
        if (time.time() - start_time) < deadline - 5.0:
            edges3, score3 = optimize_edges(values, edges, rng, deadline - 5.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': [list(e) for e in edges3]}

    # Phase 3: More random restarts with optimized structures
    if best_result is not None and (time.time() - start_time) < deadline - 25.0:
        base_values = best_result['values']
        base_edges = best_result['edges']
        
        for _ in range(15):
            if (time.time() - start_time) > deadline - 10.0:
                break
                
            # Perturb base values
            perturbed_values = list(base_values)
            idx = rng.randint(0, n-1)
            perturbed_values[idx] = rng.randint(1, 40)
            if len(set(perturbed_values)) != n:
                continue
                
            # Perturb base edges (swap 2 edges)
            perturbed_edges = [list(e) for e in base_edges]
            edges_set = set(tuple(e) for e in perturbed_edges)
            for _ in range(2):
                i = rng.randint(0, len(perturbed_edges)-1)
                new_e = rng.choice(all_edges)
                if tuple(new_e) not in edges_set:
                    old_e = perturbed_edges[i]
                    edges_set.discard(tuple(old_e))
                    perturbed_edges[i] = list(new_e)
                    edges_set.add(tuple(new_e))
            
            # Optimize
            edges_opt, s1 = optimize_edges(perturbed_values, perturbed_edges, rng, deadline - 12.0)
            values_opt, s2 = optimize_values(perturbed_values, edges_opt, rng, deadline - 8.0)
            
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values_opt), 'edges': [list(e) for e in edges_opt]}

    if best_result is None:
        # Fallback
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    # Final Validation and Formatting
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    # Check constraints
    assert len(set(values)) == 14, "Values must be distinct"
    assert all(1 <= v <= 10000 for v in values), "Values must be in range"
    assert len(edges) == 35, "Must have 35 edges"
    assert all(u != v for u, v in edges), "No loops"
    assert len(set(tuple(sorted(e)) for e in edges)) == 35, "No duplicate edges"
    assert all(0 <= u < 14 and 0 <= v < 14 for u, v in edges), "Indices in range"
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
