import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses scipy.optimize.milp to solve as an integer linear programming problem
    with exact rational arithmetic via LCM scaling.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of all denominators for exact integer arithmetic
    from math import gcd
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2, 3, ..., 48
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Scale each reciprocal: 1/d becomes L/d (integer)
    scaled_recips = [L // d for d in denominators]
    
    # We want to minimize deficit = 1 - sum(1/d_i) = (L - sum(scaled_recips[i])) / L
    # Since L is fixed, minimizing deficit is equivalent to maximizing sum(scaled_recips[i])
    # subject to sum(scaled_recips[i]) < L (strict inequality)
    # 
    # For MILP, we need: sum(scaled_recips[i]) <= L - 1 (since we need strict < L)
    #
    # Objective: minimize -sum(scaled_recips[i]) (equivalent to maximizing sum)
    # Or equivalently: minimize L - sum(scaled_recips[i]) which is the numerator of the deficit
    
    # Decision variables: x_i in {0, 1} for each denominator
    # Objective: minimize (L - sum(scaled_recips[i] * x_i))
    # This is equivalent to: minimize -sum(scaled_recips[i] * x_i) + constant L
    # So objective coefficients are: -scaled_recips[i] for each i
    
    c = np.array([-float(s) for s in scaled_recips])
    
    # Constraint: sum(scaled_recips[i] * x_i) <= L - 1
    # This ensures the sum is strictly less than L (i.e., sum of reciprocals < 1)
    
    # Also need at least one index selected (nonempty subset)
    # Constraint: sum(x_i) >= 1
    
    # MILP formulation:
    # Minimize: c^T x = -sum(scaled_recips[i] * x_i)
    # Subject to:
    #   sum(scaled_recips[i] * x_i) <= L - 1
    #   sum(x_i) >= 1
    #   x_i in {0, 1}
    
    # Linear constraints:
    # A @ x <= b
    # Row 1: sum(scaled_recips[i] * x_i) <= L - 1
    # Row 2: -sum(x_i) <= -1  (equivalent to sum(x_i) >= 1)
    
    A = np.array([
        [float(s) for s in scaled_recips],
        [-1.0] * n
    ])
    b = np.array([L - 1, -1])
    
    constraints = LinearConstraint(A, -np.inf, b)
    
    # Variable bounds: x_i in [0, 1]
    integrality = np.ones(n)  # All variables are integer (binary)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve MILP
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if time.time() > deadline:
            raise TimeoutError("MILP exceeded time limit")
        
        if result.success and result.x is not None:
            # Extract selected indices
            x = np.round(result.x).astype(int)
            indices = [i for i in range(n) if x[i] == 1]
            
            if indices:
                # Verify the solution is valid
                total = sum(scaled_recips[i] for i in indices)
                if total < L:  # Strictly below 1
                    return indices
            
            # If MILP returned a valid but empty or invalid solution, fallback
            pass
        
        # If MILP failed or returned invalid, use greedy fallback
        raise ValueError("MILP did not produce valid solution")
        
    except Exception:
        # Fallback: greedy approach
        if time.time() > deadline:
            # Return a simple valid solution
            return [0]
        
        # Greedy: try to find a good subset
        # Sort by reciprocal value (largest first) and greedily add if sum < 1
        indices = []
        cur_sum = 0
        for i in range(n):
            new_sum = cur_sum + scaled_recips[i]
            if new_sum < L:
                indices.append(i)
                cur_sum = new_sum
        
        if not indices:
            return [0]
        
        return indices

if __name__ == "__main__":
    res = construct()
    pass
