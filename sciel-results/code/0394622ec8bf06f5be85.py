import time
import math
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 80  # 80 second budget, 10s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal as Fraction
    reciprocals = [Fraction(1, d) for d in denominators]
    
    best_selected = None
    best_total = None
    best_deficit = None
    
    # Try MILP for initial solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c_float = [1.0 / d for d in denominators]
        c_obj = [-ci for ci in c_float]
        
        A = [
            c_float,
            [1.0] * n
        ]
        
        eps = 1e-9
        lb = [1, -float('inf')]
        ub = [1 - eps, float('inf')]
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = [1] * n
        bounds = Bounds(lb=[0] * n, ub=[1] * n)
        
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = [1 if xi > 0.5 else 0 for xi in res.x]
            selected = [i for i in range(n) if x[i]]
            
            if selected:
                total = sum(reciprocals[i] for i in selected)
                if total < 1:
                    best_selected = selected
                    best_total = total
                    best_deficit = 1 - total
    except Exception:
        pass
    
    # If MILP didn't give a valid result, use greedy
    if best_selected is None:
        selected = []
        total = Fraction(0, 1)
        for i in range(n):
            candidate = total + reciprocals[i]
            if candidate < 1:
                selected.append(i)
                total = candidate
        if not selected:
            return [0]
        best_selected = selected
        best_total = total
        best_deficit = 1 - total
    
    # Local search with time budget
    ls_deadline = start + 75
    selected_set = set(best_selected)
    current_total = best_total
    current_deficit = best_deficit
    
    while time.time() < ls_deadline:
        improved = False
        
        # Try adding one element
        for add_i in range(n):
            if add_i not in selected_set:
                new_total = current_total + reciprocals[add_i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < current_deficit:
                        selected_set.add(add_i)
                        current_total = new_total
                        current_deficit = new_deficit
                        improved = True
        
        # Try swapping: for each element in selected, try replacing with each element not in selected
        for rem_i in list(selected_set):
            for add_i in range(n):
                if add_i not in selected_set:
                    new_total = current_total - reciprocals[rem_i] + reciprocals[add_i]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < current_deficit:
                            selected_set.remove(rem_i)
                            selected_set.add(add_i)
                            current_total = new_total
                            current_deficit = new_deficit
                            improved = True
        
        if not improved:
            break
    
    result = list(selected_set)
    if not result:
        return [0]
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
