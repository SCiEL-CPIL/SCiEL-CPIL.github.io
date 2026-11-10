import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Candidate value sets - focus on sets with many repeated sums/products
    # Powers of 2 give unique products but varied sums
    # Small consecutive integers give many repeated sums
    # Multiples give repeated products
    
    candidates = [
        list(range(1, 15)),                          # 1-14
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],  # mixed small
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72], # multiplicative
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 40, 60], # even multiples
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], # powers of 2 with extras
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 45, 60, 90], # multiples of 3
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50], # divisors of 150
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24], # small with many factors
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # 1-14 again (redundant but safe)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], # 1-13, 15
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16], # 1-12, 14, 16
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16], # mixed
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16], # mixed
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15], # mixed
    ]
    
    # Filter valid candidates
    valid_candidates = []
    seen_sets = set()
    for v in candidates:
        if len(set(v)) != n:
            continue
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            continue
        if not all(1 <= x <= maximum for x in v):
            continue
        key = tuple(sorted(v))
        if key not in seen_sets:
            seen_sets.add(key)
            valid_candidates.append(v)
    
    if not valid_candidates:
        valid_candidates = [list(range(1, 15))]

    all_possible_edges = [list(e) for e in combinations(range(n), 2)]
    num_edges_total = len(all_possible_edges)  # 91

    def solve_for_values(vals):
        """
        Uses ILP to find the minimum K (max distinct sums/prods) for a given value set.
        Returns (score, edges) or (inf, None) if infeasible or timeout.
        """
        # Precompute sums and products for all edges
        edge_sums = [0] * num_edges_total
        edge_prods = [0] * num_edges_total
        for idx, (i, j) in enumerate(all_possible_edges):
            edge_sums[idx] = vals[i] + vals[j]
            edge_prods[idx] = vals[i] * vals[j]
        
        unique_sums = sorted(set(edge_sums))
        unique_prods = sorted(set(edge_prods))
        
        num_sums = len(unique_sums)
        num_prods = len(unique_prods)
        
        sum_index = {s: i for i, s in enumerate(unique_sums)}
        prod_index = {p: i for i, p in enumerate(unique_prods)}
        
        # ILP Formulation:
        # Minimize K
        # s.t.
        # sum(x_e) = m
        # sum(y_s) <= K
        # sum(z_p) <= K
        # x_e - y_{s_e} <= 0
        # x_e - z_{p_e} <= 0
        # x_e, y_s, z_p in {0, 1}
        
        N_x = num_edges_total
        N_y = num_sums
        N_z = num_prods
        total_vars = N_x + N_y + N_z + 1 # +1 for K
        
        # Objective: Minimize K (index total_vars - 1)
        c = np.zeros(total_vars)
        c[-1] = 1.0
        
        # Constraints matrix A, lb, ub
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. Sum(x) = m
        row = np.zeros(total_vars)
        row[:N_x] = 1.0
        A_rows.append(row)
        lb_rows.append(m)
        ub_rows.append(m)
        
        # 2. Sum(y) - K <= 0
        row = np.zeros(total_vars)
        row[N_x:N_x+N_y] = 1.0
        row[-1] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 3. Sum(z) - K <= 0
        row = np.zeros(total_vars)
        row[N_x+N_y:N_x+N_y+N_z] = 1.0
        row[-1] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 4 & 5. Linking constraints
        for e_idx in range(N_x):
            s = edge_sums[e_idx]
            p = edge_prods[e_idx]
            si = sum_index[s]
            pi = prod_index[p]
            
            # x_e - y_{s_e} <= 0
            row = np.zeros(total_vars)
            row[e_idx] = 1.0
            row[N_x + si] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
            # x_e - z_{p_e} <= 0
            row = np.zeros(total_vars)
            row[e_idx] = 1.0
            row[N_x + N_y + pi] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
        
        A = np.array(A_rows)
        lb = np.array(lb_rows)
        ub = np.array(ub_rows)
        
        constraints = LinearConstraint(A, lb, ub)
        
        integrality = np.zeros(total_vars)
        integrality[:N_x] = 1.0 # x is binary
        integrality[N_x:N_x+N_y] = 1.0 # y is binary
        integrality[N_x+N_y:N_x+N_y+N_z] = 1.0 # z is binary
        # K is continuous
        
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        bounds.ub[-1] = float('inf') # K can be large
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 5.0, 'presolve': True}
            )
            
            if res.status == 0: # Optimal
                K_val = int(np.ceil(res.fun))
                x_vals = res.x[:N_x]
                selected_indices = [i for i in range(N_x) if x_vals[i] > 0.5]
                
                if len(selected_indices) == m:
                    edges = [all_possible_edges[i] for i in selected_indices]
                    # Verify score just in case
                    actual_score = _raw_value(vals, edges)
                    return actual_score, edges
                else:
                    return float('inf'), None
            else:
                return float('inf'), None
                
        except Exception:
            return float('inf'), None

    # Iterate through candidates
    # Sort candidates by a simple heuristic: fewer unique sums/prods is better
    def heuristic_score(vals):
        s = set()
        p = set()
        for i in range(n):
            for j in range(i+1, n):
                s.add(vals[i]+vals[j])
                p.add(vals[i]*vals[j])
        return max(len(s), len(p))
        
    valid_candidates.sort(key=heuristic_score)
    
    for vals in valid_candidates:
        if time.time() > deadline - 5.0:
            break
            
        score, edges = solve_for_values(vals)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_vals = vals
            
        # Early exit if we find a very good score
        if best_score <= 10:
             break

    # Fallback if ILP failed or found nothing
    if best_edges is None:
        best_vals = valid_candidates[0]
        # Simple greedy fallback
        all_edges = all_possible_edges
        edge_sums = [best_vals[i] + best_vals[j] for i, j in all_edges]
        edge_prods = [best_vals[i] * best_vals[j] for i, j in all_edges]
        
        from collections import defaultdict
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for s in edge_sums: sum_freq[s] += 1
        for p in edge_prods: prod_freq[p] += 1
        
        scores = []
        for idx in range(len(all_edges)):
            scores.append((min(sum_freq[edge_sums[idx]], prod_freq[edge_prods[idx]]), idx))
        scores.sort(reverse=True)
        
        selected = []
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        for _, idx in scores:
            if len(selected) >= m: break
            selected.append(idx)
            sum_counts[edge_sums[idx]] += 1
            prod_counts[edge_prods[idx]] += 1
            
        best_edges = [all_edges[i] for i in selected]
        best_score = _raw_value(best_vals, best_edges)

    # Final validation
    assert len(best_vals) == n
    assert len(set(best_vals)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(best_edges) == m
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
