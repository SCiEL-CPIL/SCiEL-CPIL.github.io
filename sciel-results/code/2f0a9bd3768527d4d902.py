"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    num_possible_edges = len(all_edges)
    
    # Helper to evaluate a candidate (values, edges) using NumPy for speed
    def evaluate(values, edges):
        if not edges:
            return 0
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    def random_valid_values():
        return random.sample(range(1, max_val + 1), n)

    # Strategy: 
    # 1. Define a partition of vertices into groups.
    # 2. Within each group, try to assign values such that sums/products collide.
    #    Specifically, for a group of size k, if we can make all edges in the group
    #    produce the same sum S and same product P, that's ideal.
    #    However, for k>3, it's hard to have all pairs (u,v) in a clique have same sum and product.
    #    Example: 4 vertices a,b,c,d.
    #    Edges: ab, ac, ad, bc, bd, cd.
    #    If ab=ac => b=c. But values must be distinct.
    #    So we cannot have all edges in a clique have the same sum.
    #    
    #    Alternative: Use small sets of values that naturally collide.
    #    E.g., values {1, 2, 3, 4}.
    #    Sums: 1+2=3, 1+3=4, 1+4=5, 2+3=5, 2+4=6, 3+4=7. Distinct: 3,4,5,6,7 (5 distinct)
    #    Prods: 2,3,4,6,8,12. Distinct: 6.
    #    
    #    Let's try to find a set of 14 values that minimizes distinct sums and products
    #    for ANY 35 edges? No, we choose edges too.
    #    
    #    Better approach:
    #    1. Pick 14 values.
    #    2. Calculate sum and product for all 91 edges.
    #    3. Select 35 edges such that the number of distinct sums and products is minimized.
    #       This is a "Maximum Coverage" or "Set Cover" like problem, or simply greedy selection.
    #       Greedy: Start with empty edge set. Add edge that introduces the fewest NEW distinct sums/prods.
    #       Since we want to MINIMIZE distinct counts, we want edges that reuse existing sums/prods.
    #    
    #    This decouples the problem:
    #    A. Optimize values to have many collisions in K14.
    #    B. Optimize edges to pick the 35 "most colliding" edges.

    def select_edges_for_values(values):
        v_arr = np.array(values, dtype=np.int64)
        
        # Precompute sum and prod for all 91 edges
        # Edge index -> (u, v)
        # We want to pick 35 edges.
        # Objective: Minimize |Unique(Sums)| and |Unique(Prods)|.
        
        # Let's use a greedy heuristic:
        # Maintain a set of used sums and used prods.
        # For each step, pick the edge that adds the fewest new items to these sets.
        # Tie-breaker: Random or edge index.
        
        used_sums = set()
        used_prods = set()
        selected_edges = []
        
        # Calculate stats for all edges first? 
        # 91 edges is small.
        edge_stats = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_stats.append((u, v, s, p))
        
        # To make greedy efficient, we can just iterate.
        # 35 steps * 91 candidates * O(1) set check = ~3000 ops. Very fast.
        
        available_indices = list(range(len(edge_stats)))
        
        for _ in range(m):
            best_idx = -1
            best_cost = float('inf')
            best_new_sums = 0
            best_new_prods = 0
            
            for idx in available_indices:
                u, v, s, p = edge_stats[idx]
                
                new_s = 0 if s in used_sums else 1
                new_p = 0 if p in used_prods else 1
                
                cost = new_s + new_p
                
                # If cost is 0, it's a perfect collision.
                # If cost is 1, it adds one new item.
                # If cost is 2, it adds two new items.
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
                    best_new_sums = new_s
                    best_new_prods = new_p
                    # Early termination if we found a perfect match (cost 0)
                    if cost == 0:
                        break
            
            if best_idx == -1:
                break
            
            u, v, s, p = edge_stats[best_idx]
            selected_edges.append([u, v])
            used_sums.add(s)
            used_prods.add(p)
            available_indices.remove(best_idx)
            
        return selected_edges

    def optimize_values_local(current_values, current_edges, time_limit):
        """
        Local search on values.
        Edges are fixed.
        We want to minimize max(distinct_sums, distinct_prods).
        """
        best_vals = current_values.copy()
        best_edges = current_edges
        best_score = evaluate(best_vals, best_edges)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        while time.time() < time_limit and T > T_min:
            new_vals = best_vals.copy()
            # Perturb 1 or 2 values
            num_changes = random.randint(1, 2)
            for _ in range(num_changes):
                idx = random.randint(0, n - 1)
                # Keep value in a reasonable range to avoid huge products/sums?
                # Actually, huge values are fine as long as they are distinct.
                # But collisions are easier with small numbers.
                # Let's bias towards smaller numbers.
                # Geometric distribution or just 1..100?
                # If we use 1..100, distinctness is easier to maintain.
                # Let's sample from 1..200 to encourage collisions.
                new_val = random.randint(1, 200)
                while new_val in new_vals:
                    new_val = random.randint(1, 200)
                new_vals[idx] = new_val
            
            new_score = evaluate(new_vals, current_edges)
            
            delta = new_score - best_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                best_vals = new_vals
                best_score = new_score
            
            T *= cooling
        
        return best_vals, best_score

    def optimize_edges_local(current_values, current_edges, time_limit):
        """
        Local search on edges.
        Values are fixed.
        """
        best_edges = [list(e) for e in current_edges]
        best_score = evaluate(current_values, best_edges)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        edge_set = set(map(tuple, best_edges))
        
        while time.time() < time_limit and T > T_min:
            # Remove a random edge, add a random new edge
            rem_idx = random.randint(0, m - 1)
            rem_edge = tuple(best_edges[rem_idx])
            
            # Find a new edge
            add_edge = None
            for _ in range(20):
                cand = random.choice(all_edges)
                if cand != rem_edge and cand not in edge_set:
                    add_edge = cand
                    break
            
            if not add_edge:
                continue
            
            # Apply change
            new_edges = [list(e) for e in best_edges]
            new_edges[rem_idx] = list(add_edge)
            
            new_score = evaluate(current_values, new_edges)
            
            delta = new_score - best_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                edge_set.remove(rem_edge)
                edge_set.add(add_edge)
                best_edges = new_edges
                best_score = new_score
            
            T *= cooling
        
        return best_edges, best_score

    # Main Loop:
    # 1. Initialize with small values (1..20) to maximize collisions.
    # 2. Select best 35 edges for these values.
    # 3. Alternate between optimizing values (with fixed edges) and edges (with fixed values).
    
    best_global_score = float('inf')
    best_global_vals = None
    best_global_edges = None
    
    # Multiple restarts to avoid local minima
    num_restarts = 10
    
    for r in range(num_restarts):
        # Time check
        if time.time() > deadline - 5.0:
            break
            
        # 1. Generate initial values.
        # Use small integers to encourage collisions.
        # Distinct values from 1 to 50.
        if r == 0:
            # Specific structured start:
            # Try to make sums collide.
            # e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
            init_vals = list(range(1, 15))
        else:
            init_vals = random.sample(range(1, 100), n)
        
        # 2. Select edges greedily
        curr_edges = select_edges_for_values(init_vals)
        curr_vals = init_vals.copy()
        curr_score = evaluate(curr_vals, curr_edges)
        
        # 3. Local Search Phase 1: Optimize Values
        # Allocate time for value optimization
        t_start = time.time()
        t_limit = t_start + 3.0
        curr_vals, curr_score = optimize_values_local(curr_vals, curr_edges, t_limit)
        
        # 4. Local Search Phase 2: Optimize Edges
        # Re-select edges greedily with new values, then refine
        curr_edges = select_edges_for_values(curr_vals)
        curr_score = evaluate(curr_vals, curr_edges)
        
        t_start = time.time()
        t_limit = t_start + 3.0
        curr_edges, curr_score = optimize_edges_local(curr_vals, curr_edges, t_limit)
        
        # 5. Check if improved global best
        if curr_score < best_global_score:
            best_global_score = curr_score
            best_global_vals = curr_vals.copy()
            best_global_edges = [list(e) for e in curr_edges]
            
            # If we hit a very low score, we might stop early or do more refinement
            if best_global_score <= 5:
                # Try to refine further
                t_start = time.time()
                t_limit = t_start + 5.0
                best_global_vals, _ = optimize_values_local(best_global_vals, best_global_edges, t_limit)
                best_global_edges = select_edges_for_values(best_global_vals)
                best_global_score = evaluate(best_global_vals, best_global_edges)
                
                t_start = time.time()
                t_limit = t_start + 5.0
                best_global_edges, best_global_score = optimize_edges_local(best_global_vals, best_global_edges, t_limit)

    # Final Result
    if best_global_vals is None:
        # Fallback
        best_global_vals = list(range(1, 15))
        best_global_edges = select_edges_for_values(best_global_vals)
    
    # Ensure types are correct
    res = {
        'values': [int(v) for v in best_global_vals],
        'edges': [[int(u), int(v)] for u, v in best_global_edges]
    }
    
    return res

# EVOLVE-BLOCK-END
