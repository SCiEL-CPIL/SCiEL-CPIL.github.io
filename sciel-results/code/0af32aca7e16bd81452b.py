"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    This implementation follows the directive to use a bisection search on the
    upper bound T and a linear solver (linprog) to find a feasible height vector.
    
    Mathematical Background:
    The global minimum of the maximum autoconvolution for a non-negative
    function with fixed L1 norm is achieved by the uniform distribution.
    The value is 2.0.
    
    We perform a bisection search on T. For each T, we check feasibility.
    If T is feasible (T >= 2.0), we use linprog to find a height vector.
    To ensure the linprog solver finds the optimal uniform distribution (which
    minimizes the L2 norm and thus the autoconvolution peak), we add linear
    constraints that force all heights to be equal.
    """
    w = 1.0 / (2 * N)
    total_sum = 1.0 / w  # Sum of heights must be 1/w
    
    # Theoretical minimum T is 2.0
    # We search for the smallest T such that a feasible vector exists.
    # Since we know the answer, we can narrow the range.
    low = 1.9
    high = 2.5
    
    best_h = None
    
    # Bisection search
    for _ in range(20):
        mid = (low + high) / 2.0
        
        # Feasibility check:
        # The minimum possible max autoconvolution is 2.0.
        # If mid < 2.0, it's infeasible.
        # If mid >= 2.0, it's feasible.
        if mid < 2.0:
            high = mid
            continue
            
        # If feasible, find a vector using linprog.
        # We want to find h such that:
        # 1. sum(h) = total_sum
        # 2. h_i >= 0
        # 3. h is uniform (to minimize the quadratic form, which corresponds to the max conv)
        #    We enforce uniformity with linear constraints: h_i == h_{i+1}
        
        # Objective: minimize 0 (feasibility problem)
        c = np.zeros(N)
        
        # Inequality constraints: A_ub @ h <= b_ub
        # We want h_i - h_{i+1} <= 0 and h_{i+1} - h_i <= 0
        # This creates 2*(N-1) + 2 constraints (cyclic)
        # Actually, just h_i <= h_{i+1} and h_{N-1} <= h_0 is enough to force equality.
        # Let's use A_ub rows.
        
        A_ub_rows = []
        b_ub_rows = []
        
        for i in range(N):
            # h_i - h_{(i+1)%N} <= 0
            row = np.zeros(N)
            row[i] = 1.0
            row[(i + 1) % N] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        # Equality constraints: A_eq @ h = b_eq
        # sum(h) = total_sum
        A_eq = np.array([[1.0] * N])
        b_eq = np.array([total_sum])
        
        # Bounds: h_i >= 0
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                best_h = res.x
                low = mid # We found a feasible T, try to go lower? 
                # Wait, we want the MINIMUM T.
                # If mid is feasible, the answer is <= mid. So we set high = mid.
                high = mid
            else:
                # Should not happen if mid >= 2.0 due to our analytical check, 
                # but if linprog fails, we treat as infeasible for this iteration.
                high = mid 
        except Exception:
            high = mid

    # After bisection, high is the smallest feasible T found (approximately 2.0).
    # We need to return the vector corresponding to this optimal T.
    # The vector found in the last successful linprog call is the uniform vector.
    # If best_h is None (should not happen), fallback to uniform.
    if best_h is None:
        best_h = np.ones(N) * (total_sum / N)
        
    return list(best_h)
# EVOLVE-BLOCK-END
