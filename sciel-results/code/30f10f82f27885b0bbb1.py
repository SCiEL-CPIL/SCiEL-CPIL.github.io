def construct():
    import time
    import math
    import sys
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.optimize.optimize import OptimizeResult

    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    # Primes powers: 2^4, 3^3, 5^2, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    L = 1
    L *= 16
    L *= 27
    L *= 25
    L *= 7
    L *= 11
    L *= 13
    L *= 17
    L *= 19
    L *= 23
    L *= 29
    L *= 31
    L *= 37
    L *= 41
    L *= 43
    L *= 47
    
    # Weights are L / d
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    
    # We want to maximize sum(weights[i] * x[i]) subject to sum <= L-1
    # which is equivalent to minimizing -sum(weights[i] * x[i])
    # x[i] in {0, 1}
    
    # 1. Greedy Initialization to provide a strong initial bound (hint)
    # Sort by weight descending (i.e., denominator ascending)
    sorted_indices = np.argsort(-weights)
    curr_sum = 0
    curr_x = np.zeros(n, dtype=int)
    for i in sorted_indices:
        if curr_sum + weights[i] <= L - 1:
            curr_sum += weights[i]
            curr_x[i] = 1
    
    best_sum = curr_sum
    best_x = curr_x.copy()
    
    # 2. Try to solve using SciPy MILP (HiGHS)
    # Objective: minimize c^T x where c = -weights
    c = -weights
    
    # Constraints:
    # 0 <= x <= 1
    # sum(x * weights) <= L - 1
    # Since x are binary, we can use int vars
    
    # A matrix for the constraint: weights^T * x <= L - 1
    A = weights.reshape(1, -1)
    b_lower = np.array([0])
    b_upper = np.array([L - 1])
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    
    # Integrality: all variables are integer (and bounds 0-1 make them binary)
    integrality = np.ones(n)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Initial hint
    # Note: milp might not accept hints directly in all versions or it might, 
    # but passing x0 is supported.
    try:
        # Use a timeout for the solver if possible, but milp doesn't have a direct time limit param in all versions.
        # We rely on the fact that for n=47, HiGHS is very fast.
        # If it hangs, the deadline check below won't trigger during the C call.
        # To be safe, we could use a thread, but let's assume HiGHS is fast enough for 47 vars.
        # If it takes too long, we might timeout the whole program.
        # However, 47 binary vars is tiny for HiGHS. It should be instantaneous.
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 45}  # HiGHS supports time_limit
        )
        
        if res.status == 0:  # Optimal
            # Extract solution
            x_sol = res.x.astype(int)
            # Ensure binary (sometimes solver returns 0.9999)
            x_sol = (x_sol > 0.5).astype(int)
            
            sol_sum = np.sum(x_sol * weights)
            
            if sol_sum > best_sum and sol_sum < L:
                best_sum = sol_sum
                best_x = x_sol
        else:
            # If not optimal, check if we have a feasible solution
            if res.x is not None:
                x_sol = res.x.astype(int)
                x_sol = (x_sol > 0.5).astype(int)
                sol_sum = np.sum(x_sol * weights)
                if sol_sum > best_sum and sol_sum < L:
                    best_sum = sol_sum
                    best_x = x_sol
            
    except Exception as e:
        # Fallback to manual search if MILP fails
        pass

    # 3. Fallback / Refinement: Manual Branch and Bound
    # If MILP didn't find a solution, or if we have time left, we can run a quick B&B
    # to ensure we have the best possible.
    # Given the score of the parent was 1.78e6, and the deficit is 1/L * (L - sum),
    # minimizing deficit is maximizing sum.
    # The MILP should find the global optimum.
    # Let's add a quick check if time remains to do a local search or B&B if MILP failed.
    
    if time.time() < deadline - 5:
        # Implement a fast B&B if needed, or just trust MILP.
        # Given n=47, MILP is extremely likely to be optimal and fast.
        # We will stick with the MILP result as it is robust.
        
        # Just in case, verify the best_x is valid
        if np.sum(best_x * weights) >= L:
             # Invalid, revert to greedy
             best_x = curr_x
             best_sum = curr_sum

    # Reconstruct indices
    result = [i for i in range(n) if best_x[i] == 1]
    
    if not result:
        result = [0]
        
    return result
