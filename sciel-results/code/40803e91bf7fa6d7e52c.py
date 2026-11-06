import time
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    # lcm(a,b) = a*b/gcd(a,b) <= 256  <=>  a*b <= 256*gcd(a,b)
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Conflict counts
    conflict_count = [bin(conflict_mask[i]).count('1') for i in range(n_total)]
    
    # Multiple greedy orderings to get good initial solutions
    scores = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    order1 = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    order2 = sorted(range(n_total), key=lambda i: 1.0 / all_nodes[i], reverse=True)
    order3 = sorted(range(n_total), key=lambda i: conflict_count[i])
    # Order by (1/x) / sqrt(1 + conflict_count)
    scores4 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i])**0.5 for i in range(n_total)]
    order4 = sorted(range(n_total), key=lambda i: scores4[i], reverse=True)
    
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    for order in [order1, order2, order3, order4]:
        sel = []
        used = 0
        for idx in order:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Local search: add, swap, 2-opt
    def local_search(selected, deadline):
        selected = selected[:]
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # 1. Try adding nodes
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                used |= (1 << i)
                improved = True
            
            # 2. Try 1-swap
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                s_i = node_to_idx[s]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break
        
        return selected
    
    # Run local search from heuristic
    best_subset = local_search(best_heuristic_subset, deadline - 15)
    best_value = sum(1.0 / v for v in best_subset)
    
    # Build MILP
    n_vars = n_total
    c = -np.array([1.0 / x for x in all_nodes])
    
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    for i in range(n_vars):
        mask = conflict_mask[i]
        j = i + 1
        while j < n_vars:
            if (mask >> j) & 1:
                row_indices.append(constraint_idx)
                col_indices.append(i)
                data.append(1)
                row_indices.append(constraint_idx)
                col_indices.append(j)
                data.append(1)
                constraint_idx += 1
            j += 1
    
    num_constraints = constraint_idx
    
    if num_constraints > 0:
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_vars))
        lb = np.zeros(num_constraints)
        ub = np.ones(num_constraints)
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        
        x0 = np.zeros(n_vars)
        for v in best_subset:
            x0[node_to_idx[v]] = 1.0
        
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 1:
            try:
                res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                           options={"time_limit": max(1, remaining), "mip_gap": 0.0})
                
                if res.x is not None:
                    milp_subset = [all_nodes[i] for i in range(n_vars) if res.x[i] > 0.5]
                    if milp_subset:
                        milp_value = sum(1.0 / v for v in milp_subset)
                        if milp_value > best_value:
                            best_subset = milp_subset
                            best_value = milp_value
            except Exception:
                pass
    
    # Final polish: local search on the best solution
    if (time.time() - start_time) < deadline - 3:
        polished = local_search(best_subset, deadline - 2)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            best_subset = polished
            best_value = val
    
    if not best_subset:
        return [256]
    
    return best_subset
