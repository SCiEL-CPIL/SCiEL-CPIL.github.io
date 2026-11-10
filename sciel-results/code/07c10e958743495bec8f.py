import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    # Set a deadline guard
    start_time = time.time()
    deadline = 580  # Leave 20s margin for safety
    
    # Choose K. 
    # The current best uses K=1000 with an LP.
    # Let's try to improve by using a slightly larger K if time permits, 
    # or refine the LP solution.
    # K=1000 gives a score of ~1.0745.
    # The objective is sum_{k=2}^K -log(k)/k * f_k.
    # The constraint is sum_k f_k * floor(x/k) <= 1 for all x in [1, 10K].
    
    K = 1000
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to seed if something goes wrong immediately
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct the constraint matrix B
    # B[x-1, j-1] = floor(x/j) for x in 1..10K, j in 1..K
    # We want to minimize -c^T f subject to B f <= 1
    
    # Vectorized construction
    # x ranges from 1 to 10K
    # j ranges from 1 to K
    x_vals = np.arange(1, 10 * K + 1)[:, np.newaxis]  # shape (10K, 1)
    j_vals = np.arange(1, K + 1)[np.newaxis, :]      # shape (1, K)
    
    # Calculate floor(x/j)
    B = np.floor(x_vals / j_vals).astype(np.float64)
    
    # The constraints are B @ f <= 1
    # We also need bounds -10 <= f <= 10
    bounds = [(-10.0, 10.0)] * K
    
    # Objective: Maximize sum_{k=2}^K (-log(k)/k) * f_k
    # So minimize sum_{k=2}^K (log(k)/k) * f_k
    c = np.zeros(K)
    k_vals = np.arange(1, K + 1)
    # k=1 term is 0 in the objective sum (log(1)=0)
    c[1:] = np.log(k_vals[1:]) / k_vals[1:]
    
    # Solve LP
    # Use HiGHS if available, otherwise CBC
    try:
        res = opt.linprog(c, A_ub=B, b_ub=np.ones(10 * K), bounds=bounds, method='highs')
    except Exception:
        # If LP fails, fallback to a heuristic or seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Check time again
    if time.time() - start_time > deadline:
        # If it's taking too long, we might need to abort, but linprog is synchronous.
        # If it returns, we are good.
        pass

    if res.success:
        f_vals = res.x
    else:
        # If LP fails, fallback to a heuristic or seed
        # Try to use the seed or a simple Mobius approximation
        # For now, return seed to ensure validity
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # The LP solution maximizes the objective for k in [1, K].
    # However, the evaluator applies a transformation:
    # 1. Clip values to [-10, 10].
    # 2. S = sum_k f(k)/k over clipped values.
    # 3. f(1) = f(1) - S.
    # 4. Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K].
    # 5. Compute a = -sum_k f(k) * log(k) / k.
    
    # The LP solves for f(1)..f(K) directly.
    # But the evaluator's step 3 modifies f(1).
    # Does the LP solution remain feasible after the evaluator's modification?
    # The LP ensures sum_k f(k) floor(x/k) <= 1 for the raw f.
    # After modification, f'(1) = f(1) - S.
    # M'(x) = sum_{k=2}^K f(k) floor(x/k) + (f(1)-S) floor(x/1)
    #       = sum_{k=2}^K f(k) floor(x/k) + f(1) x - S x
    #       = M_raw(x) - S x
    # Since S = sum f(k)/k, and M_raw(x) <= 1, we have M'(x) <= 1 - Sx.
    # If S is positive, M'(x) becomes smaller, so feasibility is preserved (assuming Sx >= 0).
    # If S is negative, M'(x) could exceed 1.
    
    # Let's check S for the LP solution.
    # f_vals is the solution for k=1..K.
    # S = sum_{k=1}^K f_vals[k-1] / k
    S = np.sum(f_vals / k_vals.astype(np.float64))
    
    # If S is negative, the evaluator's modification might break feasibility.
    # In that case, we should adjust f(1) ourselves to ensure feasibility,
    # or rely on the fact that the LP solution might have S >= 0.
    
    # Let's verify the final score with the evaluator's logic.
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f_vals.astype(np.float64)
    
    # Simulate evaluator
    clipped = np.clip(values, -10, 10)
    S_eval = np.sum(clipped / domain.astype(np.float64))
    # f(1) is at index 0
    clipped[0] = clipped[0] - S_eval
    
    # Check feasibility
    feasible = True
    for x in range(1, 10 * K + 1):
        Mx = np.sum(clipped * (x // domain))
        if Mx > 1.0001:
            feasible = False
            break
    
    if not feasible:
        # If infeasible, we might need to scale down or use a different strategy.
        # For now, return the seed to be safe, or try to fix it.
        # A simple fix: if S_eval < 0, we can increase f(1) to make S_eval >= 0?
        # But f(1) is bounded by 10.
        # Let's just return the seed if infeasible to avoid score 0.
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # If feasible, return the LP solution.
    # The score is proportional to a = -sum_k f(k) log(k) / k.
    # The LP maximized this (before the f(1) adjustment, but f(1) term is 0 in objective).
    # Wait, the objective a uses the *final* values after adjustment.
    # a_final = -sum_k f'_k log(k)/k
    # f'_1 = f_1 - S
    # a_final = - (f_1 - S) log(1)/1 - sum_{k=2}^K f_k log(k)/k
    # Since log(1)=0, a_final = - sum_{k=2}^K f_k log(k)/k
    # This is exactly what the LP maximized!
    # So the LP solution is optimal for the final metric, provided feasibility holds.
    
    return domain, values
