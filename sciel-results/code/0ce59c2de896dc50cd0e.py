"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)  # 91
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def solve_ilp_for_edges(values, time_limit):
        """Use MILP to find the best 35 edges minimizing max(distinct_sums, distinct_products).
        
        Variables:
        - x[e] for e in all_edges: binary, 1 if edge e is selected
        - y_s for each possible sum s: binary, 1 if sum s appears in selected edges
        - z_p for each possible product p: binary, 1 if product p appears in selected edges
        - t: continuous, max(distinct_sums, distinct_products)
        
        Objective: minimize t
        Constraints:
        - sum(x) = m
        - For each sum s: y_s >= x[e] for all edges e with sum s
        - For each product p: z_p >= x[e] for all edges e with product p
        - t >= sum(y_s)
        - t >= sum(z_p)
        """
        # Precompute sums and products for all edges
        edge_sums = []
        edge_prods = []
        for u, v in all_edges:
            edge_sums.append(values[u] + values[v])
            edge_prods.append(values[u] * values[v])
        
        unique_sums = sorted(set(edge_sums))
        unique_prods = sorted(set(edge_prods))
        
        # Map values to indices
        sum_to_idx = {s: i for i, s in enumerate(unique_sums)}
        prod_to_idx = {p: i for i, p in enumerate(unique_prods)}
        
        ns = len(unique_sums)
        np_ = len(unique_prods)
        
        # Total variables: num_edges + ns + np_ + 1 (for t)
        num_vars = num_edges + ns + np_ + 1
        
        # Objective: minimize t (index num_edges + ns + np_)
        t_idx = num_edges + ns + np_
        c = np.zeros(num_vars)
        c[t_idx] = 1.0
        
        # Constraints:
        # 1. sum(x) = m
        # 2. For each sum s: y_s - sum_{e: sum_e = s} x_e >= 0
        # 3. For each product p: z_p - sum_{e: prod_e = p} x_e >= 0
        # 4. t - sum(y) >= 0
        # 5. t - sum(z) >= 0
        
        # Build constraint matrix
        # Constraint 1: sum of x = m
        A_rows = []
        b_lower = []
        b_upper = []
        
        # Row for sum(x) = m
        row = np.zeros(num_vars)
        row[:num_edges] = 1.0
        A_rows.append(row)
        b_lower.append(m)
        b_upper.append(m)
        
        # For each sum s: y_s >= sum of x_e where edge e has sum s
        for si, s in enumerate(unique_sums):
            row = np.zeros(num_vars)
            row[num_edges + si] = 1.0  # y_s
            for ei, es in enumerate(edge_sums):
                if es == s:
                    row[ei] = -1.0  # -x_e
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)
        
        # For each product p: z_p >= sum of x_e where edge e has product p
        for pi, p in enumerate(unique_prods):
            row = np.zeros(num_vars)
            row[num_edges + ns + pi] = 1.0  # z_p
            for ei, ep in enumerate(edge_prods):
                if ep == p:
                    row[ei] = -1.0  # -x_e
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)
        
        # t >= sum(y)
        row = np.zeros(num_vars)
        row[t_idx] = 1.0
        row[num_edges:num_edges + ns] = -1.0
        A_rows.append(row)
        b_lower.append(0)
        b_upper.append(np.inf)
        
        # t >= sum(z)
        row = np.zeros(num_vars)
        row[t_idx] = 1.0
        row[num_edges + ns:num_edges + ns + np_] = -1.0
        A_rows.append(row)
        b_lower.append(0)
        b_upper.append(np.inf)
        
        A = np.array(A_rows)
        
        # Bounds
        lb = np.zeros(num_vars)
        ub = np.ones(num_vars)
        ub[t_idx] = num_edges  # t can be at most num_edges
        
        # Integrality: x and y, z are binary; t is continuous
        integrality = np.zeros(num_vars)
        integrality[:num_edges + ns + np_] = 1  # binary
        integrality[t_idx] = 0  # continuous
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(lb, ub)
        
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_limit}
            )
            
            if result.success or result.status == 8:  # 8 = time limit reached with solution
                x_vals = result.x[:num_edges]
                selected_edges = [all_edges[i] for i in range(num_edges) if x_vals[i] > 0.5]
                
                if len(selected_edges) == m:
                    score = evaluate(values, selected_edges)
                    return selected_edges, score
                else:
                    # Fallback: take first m edges
                    selected_edges = selected_edges[:m]
                    if len(selected_edges) < m:
                        # Fill with remaining
                        for e in all_edges:
                            if e not in selected_edges:
                                selected_edges.append(e)
                            if len(selected_edges) >= m:
                                break
                    score = evaluate(values, selected_edges)
                    return selected_edges, score
            else:
                return None, None
        except Exception:
            return None, None
    
    def refine_edges_local_search(values, edges, rng, time_limit):
        """Local search on edges to minimize distinct sums and products."""
        edges_set = set(tuple(e) for e in edges)
        current_score = evaluate(values, edges)
        
        # Precompute sum and product for each edge
        edge_sp = {}
        for u, v in all_edges:
            edge_sp[(u, v)] = (values[u] + values[v], values[u] * values[v])
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(1000):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges) - 1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                new_score = evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def refine_values_local_search(values, edges, rng, time_limit):
        """Local search on values to minimize distinct sums and products."""
        current_score = evaluate(values, edges)
        value_set = set(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(200):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n - 1)
                old_val = values[idx]
                
                candidates = set()
                # Get neighbors in the edge set
                neighbor_vals = set()
                for u, v in edges:
                    if u == idx:
                        neighbor_vals.add(values[v])
                    elif v == idx:
                        neighbor_vals.add(values[u])
                
                # Try values that create sums/products already present
                existing_sums = set()
                existing_prods = set()
                for u, v in edges:
                    existing_sums.add(values[u] + values[v])
                    existing_prods.add(values[u] * values[v])
                
                for nv in neighbor_vals:
                    for es in existing_sums:
                        cand = es - nv
                        if 1 <= cand <= max_val:
                            candidates.add(cand)
                    for ep in existing_prods:
                        if nv != 0 and ep % nv == 0:
                            cand = ep // nv
                            if 1 <= cand <= max_val:
                                candidates.add(cand)
                
                for delta in range(-15, 16):
                    candidates.add(old_val + delta)
                candidates.add(rng.randint(1, max_val))
                candidates.add(rng.randint(1, 500))
                
                found_improvement = False
                for new_val in sorted(candidates):
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in value_set:
                        continue
                    values[idx] = new_val
                    value_set.discard(old_val)
                    value_set.add(new_val)
                    new_score = evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        break
                    else:
                        values[idx] = old_val
                        value_set.discard(new_val)
                        value_set.add(old_val)
                if not found_improvement:
                    values[idx] = old_val
                    value_set.add(old_val)
        
        return values, current_score
    
    # Main search
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Value strategies: try diverse initial configurations
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30],
        [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 24, 30],
        [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All same - will be filtered
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5000],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    ]
    
    # Filter out strategies with duplicate values
    valid_strategies = []
    for vs in value_strategies:
        if len(set(vs)) == n:
            valid_strategies.append(vs)
    
    for init_values in valid_strategies:
        if (time.time() - start_time) > deadline - 20.0:
            break
        
        values = list(init_values)
        
        # Step 1: Use ILP to find optimal edges for these values
        ilp_time = min(15.0, deadline - 25.0)
        if ilp_time > 1.0:
            ilp_edges, ilp_score = solve_ilp_for_edges(values, ilp_time)
            if ilp_edges is not None and ilp_score < best_overall_score:
                best_overall_score = ilp_score
                best_result = {'values': list(values), 'edges': [list(e) for e in ilp_edges]}
        
        # Step 2: Local search on edges
        if (time.time() - start_time) < deadline - 10.0:
            if best_result is not None and best_result['values'] == values:
                edges = [list(e) for e in best_result['edges']]
            else:
                # Greedy initial edges
                edges = list(all_edges[:m])
            
            edges, score2 = refine_edges_local_search(values, edges, rng, min(10.0, deadline - 10.0))
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Step 3: Local search on values
        if (time.time() - start_time) < deadline - 8.0 and best_result is not None:
            if best_result['values'] == values:
                values = list(best_result['values'])
                edges = [list(e) for e in best_result['edges']]
            else:
                continue
            
            values, score3 = refine_values_local_search(values, edges, rng, min(8.0, deadline - 8.0))
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Step 4: Another round of edge refinement
        if (time.time() - start_time) < deadline - 5.0 and best_result is not None:
            if best_result['values'] == values:
                edges = [list(e) for e in best_result['edges']]
            else:
                continue
            
            edges, score4 = refine_edges_local_search(values, edges, rng, min(5.0, deadline - 5.0))
            if score4 < best_overall_score:
                best_overall_score = score4
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}

    # Final polish on best result
    if best_result is not None and (time.time() - start_time) < deadline - 3.0:
        values = list(best_result['values'])
        edges = [list(e) for e in best_result['edges']]
        
        # Try ILP again on the best values
        ilp_time = min(10.0, deadline - 5.0)
        if ilp_time > 1.0:
            ilp_edges, ilp_score = solve_ilp_for_edges(values, ilp_time)
            if ilp_edges is not None and ilp_score < best_overall_score:
                best_overall_score = ilp_score
                best_result = {'values': list(values), 'edges': [list(e) for e in ilp_edges]}
                edges = [list(e) for e in best_result['edges']]
        
        # Alternate refinements
        for _ in range(2):
            if (time.time() - start_time) > deadline - 3.0:
                break
            values, s1 = refine_values_local_search(values, edges, rng, min(3.0, deadline - 3.0))
            edges, s2 = refine_edges_local_search(values, edges, rng, min(3.0, deadline - 3.0))
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n + 1)),
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
