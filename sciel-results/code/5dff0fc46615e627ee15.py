import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Keep a 5s safety margin
    
    target = 200
    max_val = 100
    forbidden_subsets = []
    
    # Generate all subsets of [1, 100] summing to 200
    # Using iterative stack to avoid recursion limit issues and potential overhead
    # State: (start, current_sum, current_path)
    stack = [(1, 0, [])]
    
    while stack:
        start, current_sum, current_path = stack.pop()
        
        if current_sum == target:
            if current_path:
                forbidden_subsets.append(tuple(current_path))
            continue
            
        if current_sum > target:
            continue
            
        # Pruning: if current_sum + max_possible_remaining < target, skip
        # Max possible remaining from 'start' to 'max_val'
        # Sum = (max_val + start) * (max_val - start + 1) // 2
        # If current_sum + Sum < target, we can't reach target.
        remaining_count = max_val - start + 1
        if remaining_count > 0:
            max_rem_sum = (max_val + start) * remaining_count // 2
            if current_sum + max_rem_sum < target:
                continue
        else:
            # No more numbers, but sum != target
            continue

        for i in range(start, max_val + 1):
            next_sum = current_sum + i
            if next_sum <= target:
                # We push to stack. Order doesn't matter for generation.
                # Using list append for path is O(N).
                stack.append((i + 1, next_sum, current_path + [i]))
            else:
                break
                
    if time.time() - start_time > deadline:
        return greedy_fallback()

    n_vars = 100
    n_constraints = len(forbidden_subsets)
    
    # If no forbidden subsets, return all
    if n_constraints == 0:
        return list(range(1, 101))
        
    c = np.array([-1.0] * n_vars)
    
    A = np.zeros((n_constraints, n_vars))
    b_lower = np.zeros(n_constraints)
    b_upper = np.zeros(n_constraints)
    
    for idx, subset in enumerate(forbidden_subsets):
        for val in subset:
            A[idx, val - 1] = 1.0
        b_upper[idx] = len(subset) - 1
        
    bounds = [(0, 1)] * n_vars
    integrality = np.ones(n_vars)
    
    if time.time() - start_time > deadline:
        return greedy_fallback()

    try:
        from scipy.optimize import milp, LinearConstraint
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected = [i + 1 for i in range(n_vars) if x[i] > 0.5]
            if selected:
                return selected
            else:
                return greedy_fallback()
        else:
            return greedy_fallback()
            
    except Exception:
        return greedy_fallback()

def greedy_fallback():
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
# EVOLVE-BLOCK-END
