import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
from itertools import combinations
import sys

def construct():
    n = 24
    # We will collect unique constraints (coeffs, rhs)
    # coeffs is a tuple of length 24
    # rhs is the integer |J|
    constraints_data = []
    seen_constraints = set()
    
    # For each a in 1..24, find subsets J of {a+1, ..., 24} such that sum(1/j for j in J) == 1/a
    # We use a recursive search (DFS) for each a to be efficient.
    
    def find_subsets(target, candidates):
        # candidates is a list of ints > a
        # target is Fraction
        results = []
        
        def dfs(current_sum, current_set, start_idx):
            if current_sum == target:
                if current_set:
                    results.append(tuple(current_set))
                return
            if current_sum > target:
                return
            if start_idx >= len(candidates):
                return
            
            # Option 1: Exclude candidates[start_idx]
            dfs(current_sum, current_set, start_idx + 1)
            
            # Option 2: Include candidates[start_idx]
            j = candidates[start_idx]
            new_sum = current_sum + Fraction(1, j)
            if new_sum <= target:
                current_set.append(j)
                dfs(new_sum, current_set, start_idx + 1)
                current_set.pop()
        
        dfs(Fraction(0, 1), [], 0)
        return results

    for a in range(1, 25):
        target = Fraction(1, a)
        # Candidates must be > a and <= 24
        candidates = list(range(a + 1, 25))
        if not candidates:
            continue
            
        subsets = find_subsets(target, candidates)
        
        for J in subsets:
            # J is a tuple of elements
            # Constraint: x_a + sum(x_j for j in J) <= len(J)
            coeffs = [0] * 24
            coeffs[a - 1] = 1
            for j in J:
                coeffs[j - 1] += 1
            
            key = tuple(coeffs)
            if key not in seen_constraints:
                seen_constraints.add(key)
                constraints_data.append((key, len(J)))
                
    if not constraints_data:
        return list(range(1, 25))
        
    # Build ILP
    # Maximize sum(x_i) -> Minimize -sum(x_i)
    c = np.ones(n) * -1
    
    A_ub = np.array([c for c, _ in constraints_data], dtype=float)
    b_ub = np.array([s for _, s in constraints_data], dtype=float)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    integrality = np.ones(n)
    
    # Add constraint sum(x) >= 1? No, nonempty is required by return, but ILP might return empty if no solution? 
    # The problem guarantees a nonempty subset exists (e.g., {1}).
    # Actually, if we select {1}, is it valid? 
    # 1/1 = 1. Can 1 be sum of reciprocals of others? 
    # Max sum of others is 1/2+1/3+... < 1? 
    # 1/2+1/3+1/4+1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20+1/21+1/22+1/23+1/24
    # H_24 - 1. H_24 ~ 3.77. So sum ~ 2.77 > 1.
    # So 1 CAN be represented. e.g. 1 = 1/2 + 1/3 + 1/6.
    # So {1} is NOT necessarily safe if we don't pick 2,3,6. But if we pick {1, 2, 3, 6}, it's invalid.
    # The ILP will handle this.
    
    try:
        res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], 
                   integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected = [i + 1 for i in range(n) if x[i] > 0.5]
            if selected:
                return selected
            else:
                return [1]
        else:
            # Fallback
            return list(range(13, 25))
    except Exception:
        # Fallback
        return list(range(13, 25))
