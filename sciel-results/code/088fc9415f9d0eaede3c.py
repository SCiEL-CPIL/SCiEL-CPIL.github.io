import time
from math import gcd
from fractions import Fraction
import numpy as np

def construct():
    deadline = time.time() + 55
    
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Precompute conflicts: conflict[i_idx] = set of j_idx > i_idx that conflict
    conflicts = [set() for _ in range(n_vars)]
    conflict_matrix = [set() for _ in range(n_vars)]  # symmetric
    
    for i_idx in range(n_vars):
        i = nodes[i_idx]
        for j_idx in range(i_idx + 1, n_vars):
            j = nodes[j_idx]
            if i * j <= 256 * gcd(i, j):
                conflicts[i_idx].add(j_idx)
                conflicts[j_idx].add(i_idx)
                conflict_matrix[i_idx].add(j_idx)
                conflict_matrix[j_idx].add(i_idx)
    
    def is_valid_addition(current_set, new_idx):
        """Check if new_idx can be added to current_set (a set of indices)."""
        for idx in current_set:
            if idx in conflict_matrix[new_idx]:
                return False
        return True
    
    def reciprocal_sum(indices):
        """Compute exact reciprocal sum as Fraction."""
        s = Fraction(0, 1)
        for idx in indices:
            s += Fraction(1, nodes[idx])
        return s
    
    def local_search(initial_indices, time_limit):
        """Local search: try adding, removing, and swapping to improve reciprocal sum."""
        current = set(initial_indices)
        current_sum = reciprocal_sum(current)
        
        # Sort candidates by 1/x descending (smaller x first for higher weight)
        candidates = sorted(range(n_vars), key=lambda i: -1.0 / nodes[i])
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to add any candidate
            for cand in candidates:
                if cand not in current and is_valid_addition(current, cand):
                    new_sum = current_sum + Fraction(1, nodes[cand])
                    if new_sum > current_sum:
                        current.add(cand)
                        current_sum = new_sum
                        improved = True
                        break  # restart scan after adding
            
            # Phase 2: Try swaps - remove one, add one or more
            if not improved:
                for remove_idx in list(current):
                    temp_set = current - {remove_idx}
                    temp_sum = current_sum - Fraction(1, nodes[remove_idx])
                    
                    # Try adding candidates that were previously blocked
                    for cand in candidates:
                        if cand in current:
                            continue
                        if is_valid_addition(temp_set, cand):
                            new_sum = temp_sum + Fraction(1, nodes[cand])
                            if new_sum > current_sum:
                                current = temp_set | {cand}
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                    # Also try just removing if it helps (shouldn't, but check)
        
        return list(current), current_sum
    
    # Greedy initialization: sort by 1/x descending
    greedy_indices = []
    greedy_set = set()
    for idx in sorted(range(n_vars), key=lambda i: -1.0 / nodes[i]):
        if is_valid_addition(greedy_set, idx):
            greedy_set.add(idx)
            greedy_indices.append(idx)
    
    best_indices = list(greedy_set)
    best_sum = reciprocal_sum(best_indices)
    
    # Try MILP for a potentially better initial solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        c = -np.array([1.0 / x for x in nodes])
        
        num_constraints = sum(len(cs) for cs in conflicts)
        if num_constraints > 0:
            rows = []
            cols = []
            vals = []
            constraint_idx = 0
            for i_idx, c_list in enumerate(conflicts):
                for j_idx in c_list:
                    rows.append(constraint_idx)
                    cols.append(i_idx)
                    vals.append(1)
                    rows.append(constraint_idx)
                    cols.append(j_idx)
                    vals.append(1)
                    constraint_idx += 1
            
            A = sp.csr_matrix((vals, (rows, cols)), shape=(num_constraints, n_vars))
            lb = np.zeros(num_constraints)
            ub = np.ones(num_constraints)
            constraint = LinearConstraint(A, lb, ub)
            bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
            integrality = np.ones(n_vars)
            
            res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                x = res.x
                milp_indices = [i for i in range(n_vars) if x[i] > 0.5]
                if milp_indices:
                    milp_sum = reciprocal_sum(milp_indices)
                    if milp_sum > best_sum:
                        best_indices = milp_indices
                        best_sum = milp_sum
    except Exception:
        pass
    
    # Local search refinement with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        local_deadline = time.time() + max(remaining_time - 2, 1)
        refined_indices, refined_sum = local_search(best_indices, local_deadline)
        if refined_sum > best_sum:
            best_indices = refined_indices
            best_sum = refined_sum
    
    # Convert indices back to values
    result = [nodes[idx] for idx in best_indices]
    if not result:
        result = [nodes[0]]
    
    return result
