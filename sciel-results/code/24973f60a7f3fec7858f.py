"""Strategy: Structured geometric-progression values + exact bipartite matching for edge selection."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    
    def evaluate(values, edges):
        distinct_sums = set()
        distinct_prods = set()
        for u, v in edges:
            distinct_sums.add(values[u] + values[v])
            distinct_prods.add(values[u] * values[v])
        return max(len(distinct_sums), len(distinct_prods))
    
    def select_edges_by_matching(values):
        """
        For a given value set, select m edges to minimize max(distinct_sums, distinct_prods).
        Uses a greedy + local search approach on edges, leveraging collision structures.
        """
        all_edges = list(itertools.combinations(range(n), 2))
        
        # Precompute sum and product for all edges
        edge_sum = {}
        edge_prod = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_sum[(u, v)] = s
            edge_prod[(u, v)] = p
        
        # Group edges by sum and by product
        sum_groups = defaultdict(list)
        prod_groups = defaultdict(list)
        for e in all_edges:
            sum_groups[edge_sum[e]].append(e)
            prod_groups[edge_prod[e]].append(e)
        
        # Strategy: Greedy selection prioritizing edges that create few new distinct sums/prods
        # We want to pick m edges such that the union of their sum-values and prod-values is minimized.
        # This is a set cover-like problem. We use a greedy heuristic:
        # At each step, pick the edge that minimizes the increase in (distinct_sums + distinct_prods).
        
        def greedy_select(values, all_edges, edge_sum, edge_prod, target_m):
            selected = []
            used_edges = set()
            cur_sums = set()
            cur_prods = set()
            
            remaining = list(all_edges)
            random.shuffle(remaining)  # Break ties randomly
            
            for _ in range(target_m):
                best_edge = None
                best_cost = float('inf')
                best_new_sums = None
                best_new_prods = None
                
                for e in remaining:
                    if e in used_edges:
                        continue
                    s = edge_sum[e]
                    p = edge_prod[e]
                    
                    # Cost: how many NEW distinct sums and prods does this edge add?
                    new_s = 0 if s in cur_sums else 1
                    new_p = 0 if p in cur_prods else 1
                    cost = new_s + new_p
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = e
                        best_new_sums = s
                        best_new_prods = p
                        if cost == 0:
                            break  # Can't do better than 0
                
                if best_edge is None:
                    # Fallback: pick any unused edge
                    for e in remaining:
                        if e not in used_edges:
                            best_edge = e
                            break
                
                if best_edge is not None:
                    selected.append(best_edge)
                    used_edges.add(best_edge)
                    cur_sums.add(edge_sum[best_edge])
                    cur_prods.add(edge_prod[best_edge])
                
                # Early termination if we've found enough zero-cost edges
                if len(selected) >= target_m:
                    break
            
            return selected[:target_m], cur_sums, cur_prods
        
        # Run greedy with multiple random shuffles and keep the best
        best_edges = None
        best_score = float('inf')
        
        for attempt in range(20):
            edges, sums, prods = greedy_select(values, all_edges, edge_sum, edge_prod, m)
            if len(edges) < m:
                # Fill with random edges
                for e in all_edges:
                    if e not in edges:
                        edges.append(e)
                        if len(edges) == m:
                            break
            score = max(len(sums), len(prods))
            if score < best_score:
                best_score = score
                best_edges = edges[:m]
        
        return best_edges
    
    def local_refine_edges(values, edges):
        """
        Local search on edge set: try swapping one edge for another to reduce max(distinct_sums, distinct_prods).
        """
        all_edges = list(itertools.combinations(range(n), 2))
        edge_set = set(tuple(e) for e in edges)
        
        def eval_edges(edges):
            s = set()
            p = set()
            for u, v in edges:
                s.add(values[u] + values[v])
                p.add(values[u] * values[v])
            return max(len(s), len(p))
        
        current_score = eval_edges(edges)
        improved = True
        max_iters = 500
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try removing each edge and adding a new one
            for i in range(len(edges)):
                if time.time() > deadline:
                    return edges, current_score
                
                old_edge = edges[i]
                # Try replacing with a random edge not in the set
                for _ in range(5):
                    new_edge = random.choice(all_edges)
                    if new_edge in edge_set:
                        continue
                    
                    new_edges = list(edges)
                    new_edges[i] = new_edge
                    new_score = eval_edges(new_edges)
                    
                    if new_score < current_score:
                        edges = new_edges
                        edge_set = set(tuple(e) for e in edges)
                        current_score = new_score
                        improved = True
                        break
        
        return edges, current_score
    
    # Initial values: Geometric progression for maximum product collisions
    # values[i] = base * r^i
    # With base=1, r=2: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # All products are powers of 2, so very few distinct products.
    # But sums will be many distinct values.
    
    # Alternative: Use values with a common factor to reduce both sums and products.
    # Let's try: values[i] = 2 * (3^i) for i in 0..13, but keep <= 10000.
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187, 3^8=6561, 3^9=19683 > 10000
    # So we can't go that far with pure powers.
    
    # Let's use a mix: start with powers of 2, then try local refinement on values.
    
    # Initial: powers of 2
    initial_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Check distinctness and range
    assert len(set(initial_values)) == n
    assert all(1 <= v <= 10000 for v in initial_values)
    
    current_values = list(initial_values)
    current_edges = select_edges_by_matching(current_values)
    current_score = evaluate(current_values, current_edges)
    
    best_values = list(current_values)
    best_edges = list(current_edges)
    best_score = current_score
    
    # Local refinement on values: try small perturbations
    # Use a simple hill-climbing with random restarts on values
    T = 50.0
    cooling_rate = 0.99
    min_T = 1.0
    
    all_edges = list(itertools.combinations(range(n), 2))
    
    while T > min_T and time.time() < deadline:
        new_values = list(current_values)
        
        # Mutate one value
        idx = random.randint(0, n - 1)
        if random.random() < 0.5:
            # Random replacement
            new_val = random.randint(1, 10000)
            if new_val not in new_values:
                new_values[idx] = new_val
            else:
                continue
        else:
            # Small perturbation
            delta = random.randint(-100, 100)
            new_val = new_values[idx] + delta
            if 1 <= new_val <= 10000 and new_val not in new_values:
                new_values[idx] = new_val
            else:
                continue
        
        # Re-select edges optimally for new values
        new_edges = select_edges_by_matching(new_values)
        new_score = evaluate(new_values, new_edges)
        
        # Acceptance
        delta_score = new_score - current_score
        if delta_score < 0:
            accept = True
        else:
            accept = random.random() < pow(2.71828, -delta_score / T)
        
        if accept:
            current_values = new_values
            current_edges = new_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_values = list(current_values)
                best_edges = list(current_edges)
        
        T *= cooling_rate
        
        # Periodically refine edges with local search
        if int((start_time - start_time) * 10) % 100 == 0 and time.time() < deadline - 5:
            refined_edges, refined_score = local_refine_edges(current_values, current_edges)
            if refined_score < current_score:
                current_edges = refined_edges
                current_score = refined_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(refined_edges)
    
    # Final local refinement on best solution
    if time.time() < deadline - 2:
        refined_edges, refined_score = local_refine_edges(best_values, best_edges)
        if refined_score < best_score:
            best_edges = refined_edges
            best_score = refined_score
    
    # Ensure types are correct
    vals = [int(x) for x in best_values]
    ed = [[int(u), int(v)] for u, v in best_edges]
    
    # Final validation
    assert len(vals) == n
    assert len(set(vals)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in vals)
    assert len(ed) == m
    assert all(isinstance(e, list) and len(e) == 2 for e in ed)
    edge_tuples = set(tuple(e) for e in ed)
    assert len(edge_tuples) == m
    assert all(isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n and u != v for e in ed for u, v in [e])
    
    return {'values': vals, 'edges': ed}
# EVOLVE-BLOCK-END
