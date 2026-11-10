# EVOLVE-BLOCK-START
import itertools
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = float('inf')
    best_result = None
    
    # Generate candidate values
    candidate_values = []
    
    # 1. APs
    for start in range(1, 50):
        for step in range(1, 10):
            vals = [start + i * step for i in range(14)]
            if max(vals) <= max_val and len(set(vals)) == 14:
                candidate_values.append(vals)
                
    # 2. Random small
    random.seed(42)
    for _ in range(50):
        vals = random.sample(range(1, 51), 14)
        candidate_values.append(vals)
        
    # 3. Random medium
    for _ in range(50):
        vals = random.sample(range(1, 501), 14)
        candidate_values.append(vals)
        
    # 4. Random large
    for _ in range(20):
        vals = random.sample(range(1, max_val + 1), 14)
        candidate_values.append(vals)
        
    # 5. Structured
    candidate_values.append([2**i for i in range(14)])
    candidate_values.append([i*i for i in range(1, 15)])
    candidate_values.append([i*i+i+1 for i in range(14)])
    
    # 6. Clustered
    for center in [10, 100, 1000]:
        for _ in range(10):
            vals = sorted(random.sample(range(max(1, center-10), center+11), 14))
            if len(set(vals)) == 14:
                candidate_values.append(vals)

    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for v in candidate_values:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    all_edges = list(itertools.combinations(range(n), 2))
    edge_indices = list(range(len(all_edges)))
    
    start_time = time.time()
    
    for values in unique_candidates:
        if time.time() - start_time > 50:
            break
            
        edge_sums = [values[i] + values[j] for i, j in all_edges]
        edge_prods = [values[i] * values[j] for i, j in all_edges]
        
        unique_sums = sorted(set(edge_sums))
        unique_prods = sorted(set(edge_prods))
        
        sum_to_idx = {s: i for i, s in enumerate(unique_sums)}
        prod_to_idx = {p: i for i, p in enumerate(unique_prods)}
        
        # Map each edge to its sum index and product index
        edge_sum_idx = [sum_to_idx[s] for s in edge_sums]
        edge_prod_idx = [prod_to_idx[p] for p in edge_prods]
        
        num_sums = len(unique_sums)
        num_prods = len(unique_prods)
        
        # We want to find min K such that there exists a set of 35 edges
        # with <= K distinct sums and <= K distinct products.
        
        found = False
        for K in range(1, 36):
            if time.time() - start_time > 55:
                break
                
            # Build ILP
            num_vars = len(all_edges) + num_sums + num_prods
            
            c = np.zeros(num_vars)
            
            A_ub = np.zeros((num_sums + num_prods + 2, num_vars))
            b_ub = np.zeros(num_sums + num_prods + 2)
            
            A_eq = np.zeros((1, num_vars))
            b_eq = np.array([35])
            A_eq[0, :len(all_edges)] = 1.0
            
            # 1. Sum constraints for each sum label
            for s_idx in range(num_sums):
                row_idx = s_idx
                mask = [e for e in range(len(all_edges)) if edge_sum_idx[e] == s_idx]
                A_ub[row_idx, mask] = 1.0
                y_var_idx = len(all_edges) + s_idx
                A_ub[row_idx, y_var_idx] = -K
                b_ub[row_idx] = 0.0
                
            # 2. Sum constraints for each product label
            for p_idx in range(num_prods):
                row_idx = num_sums + p_idx
                mask = [e for e in range(len(all_edges)) if edge_prod_idx[e] == p_idx]
                A_ub[row_idx, mask] = 1.0
                z_var_idx = len(all_edges) + num_sums + p_idx
                A_ub[row_idx, z_var_idx] = -K
                b_ub[row_idx] = 0.0
                
            # 3. Sum y_s <= K
            row_idx = num_sums + num_prods
            y_vars = range(len(all_edges), len(all_edges) + num_sums)
            A_ub[row_idx, y_vars] = 1.0
            b_ub[row_idx] = K
            
            # 4. Sum z_p <= K
            row_idx = num_sums + num_prods + 1
            z_vars = range(len(all_edges) + num_sums, len(all_edges) + num_sums + num_prods)
            A_ub[row_idx, z_vars] = 1.0
            b_ub[row_idx] = K
            
            integrality = np.ones(num_vars)
            bounds = Bounds(lb=0, ub=1)
            
            try:
                res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(A_eq, b_eq, b_eq)], integrality=integrality, bounds=bounds)
                if res.status == 0:
                    x = res.x[:len(all_edges)]
                    selected_edges = [i for i in range(len(all_edges)) if x[i] > 0.5]
                    
                    if len(selected_edges) == m:
                        edges = [list(all_edges[idx]) for idx in selected_edges]
                        result = {'values': list(values), 'edges': edges}
                        
                        final_sums = set(values[i] + values[j] for i, j in edges)
                        final_prods = set(values[i] * values[j] for i, j in edges)
                        final_score = max(len(final_sums), len(final_prods))
                        
                        if final_score < best_score:
                            best_score = final_score
                            best_result = result
                            found = True
                            break
            except Exception:
                pass

    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_edges[:m]]
        return {'values': values, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
