import time
from math import gcd
from functools import lru_cache
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_START = time.time()
_DEADLINE = 75.0  # Safety margin: evaluator kills at 90s

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    LCM = 232792560
    weights = np.array([LCM // k for k in range(1, 23)], dtype=np.int64)
    n_elem = 22
    indices = list(range(n_elem))
    
    # Helper to verify dissociation using numpy for speed
    def verify_dissociated(selected_indices):
        w = weights[selected_indices]
        n = len(w)
        sums = set()
        sums.add(0)
        for mask in range(1, 1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                return False
            sums.add(s)
        return True

    # Find a conflict (D, E) if S is not dissociated
    def find_conflict(selected_indices):
        w = weights[selected_indices]
        n = len(w)
        sums = {}
        for mask in range(1 << n):
            s = 0
            for b in range(n):
                if mask & (1 << b):
                    s += w[b]
            if s in sums:
                mask1 = mask
                mask2 = sums[s]
                set1 = set()
                set2 = set()
                for b in range(n):
                    if mask1 & (1 << b):
                        set1.add(selected_indices[b])
                    if mask2 & (1 << b):
                        set2.add(selected_indices[b])
                D = set1 - set2
                E = set2 - set1
                if D and E:
                    return (frozenset(D), frozenset(E))
            else:
                sums[s] = mask
        return None

    # Precompute small conflicts for initial MILP constraints
    small_conflicts = set()
    
    # Check w_i == w_j + w_k
    for i in range(n_elem):
        for j in range(n_elem):
            if i == j: continue
            for k in range(n_elem):
                if i == k or j == k: continue
                if weights[i] == weights[j] + weights[k]:
                    D = frozenset([i])
                    E = frozenset([j, k])
                    small_conflicts.add((D, E))
    
    # Check w_i + w_j == w_k + w_l
    for i in range(n_elem):
        for j in range(i+1, n_elem):
            s1 = weights[i] + weights[j]
            for k in range(n_elem):
                if k == i or k == j: continue
                for l in range(k+1, n_elem):
                    if l == i or l == j: continue
                    s2 = weights[k] + weights[l]
                    if s1 == s2:
                        D = frozenset([i, j])
                        E = frozenset([k, l])
                        if D.isdisjoint(E):
                            small_conflicts.add((D, E))

    # MILP Solver - simplified
    def solve_milp(conflicts):
        if not conflicts:
            return list(range(n_elem))
        
        n_vars_x = n_elem
        n_vars_y = 2 * len(conflicts)
        n_vars = n_vars_x + n_vars_y
        
        c = np.zeros(n_vars)
        c[:n_vars_x] = -1.0
        
        A_rows = []
        lb_list = []
        ub_list = []
        
        for idx, (D, E) in enumerate(conflicts):
            y_D_idx = n_vars_x + 2*idx
            y_E_idx = n_vars_x + 2*idx + 1
            
            # y_D + y_E <= 1
            row = np.zeros(n_vars)
            row[y_D_idx] = 1
            row[y_E_idx] = 1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1)
            
            # y_D <= x_i for i in D
            for i in D:
                row = np.zeros(n_vars)
                row[i] = -1
                row[y_D_idx] = 1
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0)
                
            # y_D >= sum(x_i) - (|D|-1)
            row = np.zeros(n_vars)
            for i in D:
                row[i] = 1
            row[y_D_idx] = -1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(len(D) - 1)
            
            # y_E <= x_j for j in E
            for j in E:
                row = np.zeros(n_vars)
                row[j] = -1
                row[y_E_idx] = 1
                A_rows.append(row)
                lb_list.append(-np.inf)
                ub_list.append(0)
                
            # y_E >= sum(x_j) - (|E|-1)
            row = np.zeros(n_vars)
            for j in E:
                row[j] = 1
            row[y_E_idx] = -1
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(len(E) - 1)
            
        if not A_rows:
            return list(range(n_elem))
            
        A = np.array(A_rows)
        lb = np.array(lb_list)
        ub = np.array(ub_list)
        
        constraint = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_vars)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x[:n_vars_x]
                selected = [i for i in range(n_elem) if x[i] > 0.5]
                return selected
        except:
            pass
        return None

    # Local search: try to add elements to improve
    def local_improve(current_set, max_iter=50):
        if not _time_ok():
            return current_set
        current = list(current_set)
        current_sums = set()
        for mask in range(1 << len(current)):
            s = 0
            for b in range(len(current)):
                if mask & (1 << b):
                    s += weights[current[b]]
            current_sums.add(s)
        
        improved = True
        while improved and _time_ok():
            improved = False
            for i in range(n_elem):
                if i in current:
                    continue
                w = weights[i]
                valid = True
                for s in current_sums:
                    if s + w in current_sums:
                        valid = False
                        break
                if valid:
                    current.append(i)
                    new_sums = set()
                    for s in current_sums:
                        new_sums.add(s + w)
                    current_sums = current_sums | new_sums
                    improved = True
                    break
        return current

    best_set = None
    best_size = 0
    
    # Multiple greedy strategies
    strategies = [
        sorted(indices, key=lambda i: weights[i], reverse=True),
        sorted(indices, key=lambda i: weights[i]),
        sorted(indices, key=lambda i: -np.log1p(weights[i])),
        random.sample(indices, n_elem)
    ]
    
    for order in strategies:
        if not _time_ok():
            break
        current_set = []
        current_sums = {0}
        for i in order:
            if not _time_ok():
                break
            w = weights[i]
            new_sums = set()
            valid = True
            for s in current_sums:
                if s + w in current_sums:
                    valid = False
                    break
                new_sums.add(s + w)
            if valid:
                current_sums = current_sums | new_sums
                current_set.append(i)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
    
    # Local improve
    if best_set:
        improved = local_improve(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # MILP Iterative Improvement
    conflicts = list(small_conflicts)
    
    for iter in range(15):
        if not _time_ok():
            break
        
        sol = solve_milp(conflicts)
        if sol is None:
            break
            
        # Try to locally improve the MILP solution
        sol_improved = local_improve(sol)
        
        if verify_dissociated(sol_improved):
            if len(sol_improved) > best_size:
                best_set = sol_improved
                best_size = len(sol_improved)
            break
        
        # If invalid, find conflict and add
        conflict = find_conflict(sol_improved)
        if conflict is None:
            break
            
        D_f, E_f = conflict
        pair = (D_f, E_f)
        if pair not in conflicts:
            conflicts.append(pair)
        
        # Also try adding the reverse
        pair_rev = (E_f, D_f)
        if pair_rev not in conflicts:
            conflicts.append(pair_rev)
    
    # Final fallback if MILP didn't improve
    if not best_set:
        best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]
    
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
        
    return result
