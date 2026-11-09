import time
from math import lcm, gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    # Numbers in range [2, 256]
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Map number to index
    num_to_idx = {num: i for i, num in enumerate(numbers)}
    
    # Weights are reciprocals
    weights = np.array([1.0 / num for num in numbers])
    
    # Build conflict graph
    # Edge (i, j) exists if lcm(numbers[i], numbers[j]) <= 256
    # We need constraints x_i + x_j <= 1 for all such pairs
    
    constraints_A_rows = []
    constraints_b_ub = []
    
    # Iterate over all pairs
    # Optimization: Only check pairs where it's possible for lcm <= 256
    # If a > 256, impossible (but max is 256)
    # If a * b > 256 * gcd(a,b), then lcm > 256.
    # We can just iterate all pairs i < j and check lcm.
    # n=255, pairs ~ 32k. This is very fast.
    
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            # Quick check: if a*b > 256 * 256, lcm is definitely > 256? No.
            # lcm(a,b) = a*b/gcd(a,b).
            # If a*b > 256 * 256, it doesn't help directly.
            # Just compute lcm.
            l = lcm(a, b)
            if l <= 256:
                # Add constraint x_i + x_j <= 1
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                constraints_A_rows.append(row)
                constraints_b_ub.append(1)
                
                # Time check inside loop just in case n was larger
                if time.time() - start_time > DEADLINE:
                     # Fallback: Greedy
                     return _greedy_fallback(numbers)

    if not constraints_A_rows:
        # No conflicts, take all
        return numbers

    A = np.array(constraints_A_rows)
    b_ub = np.array(constraints_b_ub)
    
    # Bounds for variables: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers (binary)
    integrality = np.ones(n)
    
    # Objective: Minimize -sum(weights * x)
    c = -weights
    
    constraint = LinearConstraint(A, ub=b_ub)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            # Extract solution
            x = res.x
            # Due to numerical precision, threshold at 0.5
            selected_indices = np.where(x > 0.5)[0]
            selected_numbers = [numbers[i] for i in selected_indices]
            
            if not selected_numbers:
                # Fallback to greedy if solver returns empty for some reason
                return _greedy_fallback(numbers)
                
            return selected_numbers
        else:
            # If solver fails, use greedy
            return _greedy_fallback(numbers)
            
    except Exception as e:
        # On any error, use greedy
        return _greedy_fallback(numbers)

def _greedy_fallback(numbers):
    """
    Simple greedy fallback: sort by weight (1/x) descending (i.e., x ascending),
    pick if no conflict with already picked.
    Note: Greedy by weight is not optimal, but provides a valid subset.
    """
    # Sort by reciprocal weight descending -> x ascending
    # Actually, larger weight means smaller x.
    # Let's just iterate and check.
    # A better greedy: sort by "value" (1/x) desc, then pick if compatible.
    sorted_nums = sorted(numbers, key=lambda x: -1.0/x)
    
    selected = []
    for num in sorted_nums:
        valid = True
        for s in selected:
            if lcm(num, s) <= 256:
                valid = False
                break
        if valid:
            selected.append(num)
    
    if not selected:
        return [2] # Ensure non-empty
        
    return selected
