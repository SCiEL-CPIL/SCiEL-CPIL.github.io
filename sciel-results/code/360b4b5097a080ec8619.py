"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = 256
    
    # 1. Generate all 3-term APs (a, b, c) with 1 <= a < b < c <= n
    ap_constraints = []
    for a in range(1, n + 1):
        # b must be > a and c = 2b - a must be <= n
        # 2b - a <= n => b <= (n + a) / 2
        max_b = (n + a) // 2
        for b in range(a + 1, max_b + 1):
            c = 2 * b - a
            if c <= n:
                # Indices are 0-based for the vector x
                # a, b, c are 1-based
                ap_constraints.append((a - 1, b - 1, c - 1))
    
    m = len(ap_constraints)
    
    # 2. Define the MILP problem
    # Variables x_i for i in 1..n
    c_obj = np.array([1.0 / i for i in range(1, n + 1)])
    
    # Constraints: x_a + x_b + x_c <= 2 for each AP
    # A_ub x <= b_ub
    A_ub = np.zeros((m, n))
    for idx, (a, b, c) in enumerate(ap_constraints):
        A_ub[idx, a] = 1
        A_ub[idx, b] = 1
        A_ub[idx, c] = 1
    b_ub = np.full(m, 2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer
    integrality = np.ones(n)
    
    # 3. Solve
    best_result = None
    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': DEADLINE - (time.time() - start_time) - 2.0} # Leave 2s for cleanup
        )
        
        if res.status == 0 or res.status == 1: # Optimal or best feasible
            # res.x contains the solution
            # Due to floating point precision, we threshold
            x_sol = res.x
            # Filter small numerical errors
            x_sol[x_sol < 0.5] = 0
            x_sol[x_sol >= 0.5] = 1
            
            selected_set = set()
            for i, val in enumerate(x_sol):
                if val == 1:
                    selected_set.add(i + 1)
            
            if selected_set:
                best_result = list(selected_set)
    except Exception:
        pass
    
    # Fallback if MILP failed or returned empty
    if not best_result:
        # Simple greedy fallback
        selected = set()
        # Precompute APs for faster check? Or just check on fly.
        # On fly is O(|S|) per candidate. |S| is small.
        for x in range(1, n + 1):
            valid = True
            for a in selected:
                # Check if (a, x, c) is AP => c = 2x - a
                c = 2 * x - a
                if c in selected and c != x and c != a:
                    valid = False
                    break
                # Check if (a, b, x) is AP => b = (a+x)/2
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if b in selected and b != x and b != a:
                        valid = False
                        break
                # Check if (x, a, c) is AP => c = 2a - x
                c = 2 * a - x
                if c in selected and c != x and c != a:
                    valid = False
                    break
            if valid:
                selected.add(x)
        best_result = list(selected)
        
    return best_result
# EVOLVE-BLOCK-END
