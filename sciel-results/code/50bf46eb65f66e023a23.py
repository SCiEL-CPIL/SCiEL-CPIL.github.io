import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import sys

def construct():
    n = 24
    valid_constraints_data = []
    seen = set()
    
    # For each a in 1..24, find subsets J of {a+1, ..., 24} such that sum(1/j for j in J) == 1/a
    # Logic: If j < a, 1/j > 1/a, so sum > 1/a. Thus J must be subset of {a+1, ..., 24}.
    for a in range(1, 25):
        available_elements = list(range(a + 1, 25))
        if not available_elements:
            continue
            
        target = Fraction(1, a)
        
        # Precompute suffix sums for pruning
        # suffix_sum[i] = sum(1/available_elements[j] for j in range(i, len(available_elements)))
        suffix_sum = [Fraction(0,1)] * (len(available_elements) + 1)
        for i in range(len(available_elements) - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + Fraction(1, available_elements[i])
        
        def dfs_opt(idx, current_sum, current_set):
            if current_sum > target:
                return
            if idx == len(available_elements):
                if current_sum == target and current_set:
                    coeffs = [0] * 24
                    coeffs[a-1] = 1
                    for j in current_set:
                        coeffs[j-1] += 1
                    key = tuple(coeffs)
                    if key not in seen:
                        seen.add(key)
                        valid_constraints_data.append((coeffs, len(current_set)))
                return
            
            # Pruning: if current_sum + suffix_sum[idx] < target, we can't reach target even with all remaining
            if current_sum + suffix_sum[idx] < target:
                return

            # Option 1: Don't pick
            dfs_opt(idx + 1, current_sum, current_set)
            
            # Option 2: Pick
            dfs_opt(idx + 1, current_sum + Fraction(1, available_elements[idx]), current_set + [available_elements[idx]])
            
        dfs_opt(0, Fraction(0,1), [])

    m = len(valid_constraints_data)
    if m == 0:
        return list(range(1, 25))
        
    A_ub = np.array([c for c, _ in valid_constraints_data], dtype=float)
    b_ub = np.array([s for _, s in valid_constraints_data], dtype=float)
    
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = np.ones(n) * -1
    
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub)]
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x = res.x
        selected = [i+1 for i in range(n) if x[i] > 0.5]
        if selected:
            return selected
        else:
            return [1]
    else:
        return list(range(13, 25))
