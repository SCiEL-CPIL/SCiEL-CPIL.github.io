def construct():
    import time
    import math
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2 to 48
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L / d
    weights = np.array([L // d for d in denominators])
    
    # Objective: Maximize sum of weights, so minimize -sum of weights
    c = -weights
    
    # Constraints:
    # 1. Sum of weights * x <= L - 1 (strictly less than L)
    # 2. 0 <= x <= 1
    A_ub = weights.reshape(1, -1)
    b_ub = np.array([L - 1])
    
    # Using LinearConstraint for scipy.optimize.milp
    # Lower bound: 0
    # Upper bound: L - 1
    constraints = LinearConstraint(A_ub, 0, L - 1)
    
    # Integrality: 1 means integer (binary since bounds are 0,1)
    integrality = np.ones(n)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        # Check if solution is valid
        if res.status == 0 and res.fun is not None:
            # Extract solution
            x = res.x
            # Threshold to handle floating point precision
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            
            if len(selected_indices) > 0:
                # Verify strictly less than 1
                # Sum of reciprocals
                total_recip = sum(1.0 / denominators[i] for i in selected_indices)
                
                # If due to floating point it's >= 1, we might need to adjust,
                # but ILP constraint was integer sum < L, so it should be safe.
                # However, let's double check with fractions to be absolutely sure
                from fractions import Fraction
                exact_sum = sum(Fraction(1, denominators[i]) for i in selected_indices)
                
                if exact_sum < 1:
                    return selected_indices
                else:
                    # Fallback: remove one item if it's too close (should not happen with L-1 constraint)
                    # Try removing the largest weight item
                    if selected_indices:
                        selected_indices.pop(0)
                        return selected_indices if selected_indices else [0]
                    else:
                        return [0]
            else:
                # No items selected? Return smallest denominator index
                return [0]
        else:
            # Fallback to greedy if MILP fails
            best_sum = 0
            best_indices = []
            current_sum = 0
            current_indices = []
            for i in range(n):
                if current_sum + weights[i] < L:
                    current_sum += weights[i]
                    current_indices.append(i)
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices
            return best_indices if best_indices else [0]
            
    except Exception:
        # Fallback to greedy if MILP fails
        best_sum = 0
        best_indices = []
        current_sum = 0
        current_indices = []
        for i in range(n):
            if current_sum + weights[i] < L:
                current_sum += weights[i]
                current_indices.append(i)
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = current_indices
        return best_indices if best_indices else [0]
