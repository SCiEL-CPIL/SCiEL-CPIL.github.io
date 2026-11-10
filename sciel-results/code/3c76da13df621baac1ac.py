import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    start_time = time.time()
    timeout = 50  # safety margin before 90s kill
    
    # Known valid set: [66, 100] has 35 elements, no subset sums to 200
    # (max pair sum = 99+100 = 199 < 200, min triple sum = 66+67+68 = 201 > 200)
    best = list(range(66, 101))
    
    # Try ILP on [40, 100] to find larger valid subset
    L = 40
    R = 100
    elements = list(range(L, R + 1))
    n = len(elements)
    idx_map = {val: i for i, val in enumerate(elements)}
    
    # Find max k such that min k-sum <= 200
    max_k = 1
    while max_k < n:
        min_sum = max_k * L + max_k * (max_k - 1) // 2
        if min_sum > 200:
            max_k -= 1
            break
        max_k += 1
    
    # Generate constraints: for each k, find all k-subsets summing to 200
    # For each such subset, we can select at most k-1 elements
    constraints = []
    constraint_k = []  # track k for each constraint
    
    for k in range(2, max_k + 1):
        for comb in itertools.combinations(elements, k):
            if sum(comb) == 200:
                row = [0] * n
                for val in comb:
                    row[idx_map[val]] = 1
                constraints.append(row)
                constraint_k.append(k)
        
        # Time check
        if time.time() - start_time > timeout / 3:
            break
    
    if time.time() - start_time > timeout / 2:
        return best
    
    if not constraints:
        # No constraints means all elements can be selected
        return elements

    # Build ILP with per-constraint RHS
    c = -np.ones(n)
    A_ub = np.array(constraints)
    b_ub = np.array([k - 1 for k in constraint_k], dtype=float)
    
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], 
                   integrality=integrality, bounds=bounds,
                   options={'time_limit': 30})
        
        if res.status == 0:
            selected_indices = np.where(res.x > 0.5)[0]
            selected_values = [elements[i] for i in selected_indices]
            
            # Quick verification using subset sum check
            def has_subset_sum_200(vals):
                # Use DP to check if any subset sums to 200
                reachable = {0}
                for v in vals:
                    new_reachable = set()
                    for s in reachable:
                        if s + v <= 200:
                            new_reachable.add(s + v)
                    reachable |= new_reachable
                    if 200 in reachable:
                        return True
                return False
            
            if not has_subset_sum_200(selected_values) and len(selected_values) > len(best):
                best = selected_values
    except Exception:
        pass
    
    return best
