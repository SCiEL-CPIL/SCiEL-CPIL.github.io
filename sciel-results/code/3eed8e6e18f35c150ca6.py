import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    # Domain: 2 to 256
    N = 256
    # Variables x_i for i in 2..256
    # Index i in array corresponds to number i+2? Or just map number to index.
    # Let's use indices 0..254 corresponding to numbers 2..256.
    
    # Objective: Maximize sum(1/i * x_i)
    # milp minimizes c @ x. So c = -1/i.
    c = np.array([-1.0 / (i + 2) for i in range(N - 1)])
    
    # Constraints:
    # For each L in 1..256, sum_{d|L, d>=2} x_{d-2} <= 1
    # Note: if L=1, no d>=2 divides 1. Sum is 0. 0<=1 is always true.
    # We can skip L=1.
    
    A_list = []
    b_ub_list = []
    
    for L in range(2, N + 1):
        row = np.zeros(N - 1)
        count = 0
        for d in range(2, N + 1):
            if L % d == 0:
                # d is a divisor of L
                # x_d is at index d - 2
                row[d - 2] = 1.0
                count += 1
        if count > 0:
            A_list.append(row)
            b_ub_list.append(1.0)
            
    if not A_list:
        # Should not happen
        return [2]
        
    A = np.array(A_list)
    b_ub = np.array(b_ub_list)
    
    # Variable bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 for all variables (binary)
    integrality = np.ones(N - 1)
    
    constraints = LinearConstraint(A, ub=b_ub)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback if solver fails
        # Greedy: pick smallest numbers that don't conflict?
        # Actually, just return empty or single element?
        # Better fallback: simple greedy
        selected = []
        for x in range(2, N + 1):
            valid = True
            for L in range(2, N + 1):
                if L % x == 0:
                    # Check if any already selected y divides L?
                    # No, the constraint is: if we pick x, we cannot pick any y such that lcm(x,y) <= 256.
                    # This is equivalent to: for all L <= 256, if x|L, then no other selected y|L.
                    # Simpler check for greedy:
                    pass
            
            # Let's just use the property: conflict if lcm(a,b) <= 256
            # This is O(N^2)
            from math import lcm
            ok = True
            for y in selected:
                if lcm(x, y) <= 256:
                    ok = False
                    break
            if ok:
                selected.append(x)
        return selected

    x_sol = res.x
    selected = []
    for i in range(N - 1):
        if x_sol[i] > 0.5:
            selected.append(i + 2)
            
    return selected
