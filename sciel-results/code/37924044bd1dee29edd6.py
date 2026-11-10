import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 50  # 10 seconds safety margin before 60s kill

    def greedy_fallback():
        """Greedy from top down, avoiding subset sum 200."""
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    # Try multiple lower bounds L for the interval [L, 100]
    # Larger L means fewer elements but potentially denser valid subsets
    # We want to maximize cardinality, so we try various L values
    
    best_result = greedy_fallback()
    best_card = len(best_result)
    
    # Try ILP approach for different L values
    # For L=40, elements are 40..100 (61 elements)
    # For L=50, elements are 50..100 (51 elements)
    # For L=60, elements are 60..100 (41 elements)
    # For L=70, elements are 70..100 (31 elements)
    # For L=80, elements are 80..100 (21 elements)
    
    for L in [40, 50, 60, 70, 80, 90]:
        if time.time() - start_time > timeout:
            break
            
        elements = list(range(L, 101))
        n = len(elements)
        idx_map = {val: i for i, val in enumerate(elements)}
        
        # If this interval is smaller than our best, skip
        if n <= best_card:
            continue
            
        # Generate constraints: subsets summing to 200
        # Max size k_max such that min sum <= 200
        # min sum of k elements is k*L + k*(k-1)/2
        max_k = 1
        while max_k < n:
            min_sum = max_k * L + max_k * (max_k - 1) // 2
            if min_sum > 200:
                max_k -= 1
                break
            max_k += 1
        
        # We need to check k from 2 to max_k
        # Pairs: two elements summing to 200
        # For L=40: pairs like (100,100) not possible since distinct, (99,101) out of range
        # Actually for distinct elements in [L,100], max pair sum is 100+99=199 < 200
        # So no pairs can sum to 200! Start from k=3
        
        constraints = []
        
        # Generate all combinations of size 3 to max_k that sum to 200
        for k in range(3, max_k + 1):
            for comb in itertools.combinations(elements, k):
                if sum(comb) == 200:
                    # Create constraint row
                    row = [0] * n
                    for val in comb:
                        row[idx_map[val]] = 1
                    constraints.append(row)
            
            # Time check
            if time.time() - start_time > timeout:
                break
                
        if time.time() - start_time > timeout:
            break
            
        # If no constraints found, we can take all elements
        if not constraints:
            return elements
        
        # Build ILP
        # Maximize sum(x)
        # c = -1 for all (minimize -sum(x))
        c = -np.ones(n)
        
        A_ub = np.array(constraints)
        b_ub = np.ones(len(constraints)) * 2  # Sum <= 2 (at most 2 from each forbidden subset)
        
        # x_i in {0, 1}
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        try:
            res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], 
                      integrality=integrality, bounds=bounds,
                      options={"time_limit": 20})
            
            if res.status == 0:
                # res.x is the solution
                selected_indices = np.where(res.x > 0.5)[0]
                selected_values = [elements[i] for i in selected_indices]
                
                if len(selected_values) > best_card:
                    best_result = selected_values
                    best_card = len(selected_values)
            else:
                # ILP failed, try greedy on this interval
                selected, reachable = [], {0}
                for x in reversed(elements):
                    new = {s + x for s in reachable if s + x <= 200}
                    if 200 not in new:
                        selected.append(x)
                        reachable |= new
                if len(selected) > best_card:
                    best_result = selected
                    best_card = len(selected)
        except Exception:
            # Fallback to greedy on this interval
            selected, reachable = [], {0}
            for x in reversed(elements):
                new = {s + x for s in reachable if s + x <= 200}
                if 200 not in new:
                    selected.append(x)
                    reachable |= new
            if len(selected) > best_card:
                best_result = selected
                best_card = len(selected)

    return best_result
