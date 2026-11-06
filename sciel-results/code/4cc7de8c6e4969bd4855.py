import time
import random
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
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
    
    # Dominance pruning: if a > b and conflicts(a) ⊆ conflicts(b), then b dominates a
    # We only keep non-dominated nodes
    dominated = [False] * n_total
    for i in range(n_total):
        if dominated[i]:
            continue
        for j in range(n_total):
            if i == j or dominated[j]:
                continue
            # j dominates i if all_nodes[j] < all_nodes[i] and conflict_mask[j] ⊆ conflict_mask[i]
            if all_nodes[j] < all_nodes[i] and (conflict_mask[j] | conflict_mask[i]) == conflict_mask[i]:
                dominated[i] = True
    
    # Keep only non-dominated nodes for search, but we still need to consider all for final answer
    # Actually, let's just use all nodes but be smarter about search
    
    # Multiple greedy orderings to get good initial solutions
    scores1 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    order1 = sorted(range(n_total), key=lambda i: scores1[i], reverse=True)
    order2 = sorted(range(n_total), key=lambda i: 1.0 / all_nodes[i], reverse=True)
    order3 = sorted(range(n_total), key=lambda i: conflict_count[i])
    scores4 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i])**0.5 for i in range(n_total)]
    order4 = sorted(range(n_total), key=lambda i: scores4[i], reverse=True)
    # New: order by (1/x) / (1 + conflict_count)^0.75
    scores5 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i])**0.75 for i in range(n_total)]
    order5 = sorted(range(n_total), key=lambda i: scores5[i], reverse=True)
    
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    all_orders = [order1, order2, order3, order4, order5]
    
    for order in all_orders:
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
    
    # Also try some random restarts
    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(range(n_total))
        rng.shuffle(shuffled)
        # Sort by 1/x with some randomness
        shuffled.sort(key=lambda i: 1.0 / all_nodes[i] * rng.uniform(0.8, 1.2), reverse=True)
        sel = []
        used = 0
        for idx in shuffled:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Enhanced local search with 2-swap
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
            swap_done = False
            for s_idx in range(len(selected)):
                s_i = node_to_idx[selected[s_idx]]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= selected[s_idx]:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / selected[s_idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    swap_done = True
                    break
            
            # 3. Try 2-swap (if no 1-swap found improvement)
            if not swap_done and not improved:
                # Try swapping two selected with two non-selected
                if len(selected) >= 2:
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    best_gain = 0
                    best_swap = None
                    for i in range(len(selected)):
                        for j in range(i + 1, len(selected)):
                            si = node_to_idx[selected[i]]
                            sj = node_to_idx[selected[j]]
                            used_without_two = used & ~(1 << si) & ~(1 << sj)
                            cur_gain_base = 1.0 / all_nodes[si] + 1.0 / all_nodes[sj]
                            for ni in non_selected:
                                if used_without_two & conflict_mask[ni]:
                                    continue
                                for nj in non_selected:
                                    if ni == nj:
                                        continue
                                    if used_without_two & conflict_mask[nj]:
                                        continue
                                    if conflict_mask[ni] & (1 << nj):
                                        continue
                                    new_gain = 1.0 / all_nodes[ni] + 1.0 / all_nodes[nj]
                                    gain = new_gain - cur_gain_base
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_swap = (i, j, ni, nj)
                            if best_swap is not None and best_swap[0] == i and best_swap[1] == j:
                                break
                        if best_swap is not None and best_swap[0] == i:
                            break
                    if best_swap is not None:
                        i, j, ni, nj = best_swap
                        si = node_to_idx[selected[i]]
                        sj = node_to_idx[selected[j]]
                        selected[i] = all_nodes[ni]
                        selected[j] = all_nodes[nj]
                        used = used & ~(1 << si) & ~(1 << sj) | (1 << ni) | (1 << nj)
                        improved = True
        
        return selected
    
    # Run local search from heuristic
    best_subset = local_search(best_heuristic_subset, deadline - 20)
    best_value = sum(1.0 / v for v in best_subset)
    
    # Try local search from a few other starting points
    for order in all_orders[:3]:
        if (time.time() - start_time) > deadline - 25:
            break
        sel = []
        used = 0
        for idx in order:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        polished = local_search(sel, deadline - 22)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            best_subset = polished
            best_value = val
    
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
