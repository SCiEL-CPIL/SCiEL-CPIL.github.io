# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses a cutting-plane LP approach:
    - Objective: maximize a = -sum_k f(k) * log(k) / k
    - Constraints: M(x) = sum_k f(k) * floor(x/k) <= 1 for x in [1, 10K]
    - Also: sum_k f(k)/k = 0 (handled by evaluator, but we can include it)
    - Values clipped to [-10, 10]

    Strategy:
    1. Choose domain D = {1, 2, ..., K} for some K
    2. Start with coarse grid of x values
    3. Solve LP
    4. Check all x in [1, 10K] for violations
    5. Add violated constraints, repeat
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Choose domain size - start with moderate K
    # The objective benefits from including more small k values
    # K=50 gives 500 x values to check, manageable
    K = 50
    domain = np.arange(1, K + 1, dtype=np.int64)
    m = len(domain)
    
    # Precompute log(k)/k for objective
    log_k_over_k = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
    # Objective: maximize -sum f(k) * log(k)/k = minimize sum f(k) * log(k)/k
    c = log_k_over_k.copy()
    
    # Precompute floor(x/k) for all x in [1, 10K]
    max_x = 10 * K
    x_all = np.arange(1, max_x + 1, dtype=np.int64)
    
    # floor(x/k) matrix: shape (max_x, m)
    # A[x_idx, k_idx] = floor((x_idx+1) / (k_idx+1))
    x_vals = x_all.astype(np.float64)
    k_vals = domain.astype(np.float64)
    A_full = np.floor(np.outer(x_vals, 1.0 / k_vals)).astype(np.float64)
    
    # Initial coarse grid: every 5th x value, plus x=1
    coarse_step = 5
    coarse_idx = np.arange(0, max_x, coarse_step)
    if coarse_idx[-1] != max_x - 1:
        coarse_idx = np.concatenate([coarse_idx, [max_x - 1]])
    
    # Start with coarse constraints
    active_x_idx = set(coarse_idx.tolist())
    
    best_result = None
    best_objective = -np.inf
    
    # Boundary constraints: f(k) <= 10, f(k) >= -10
    # In linprog: A_ub @ x <= b_ub
    # f(k) <= 10: row of 1s at position k
    # f(k) >= -10: -f(k) <= 10: row of -1 at position k
    
    # We'll build the constraint matrix incrementally
    # Start with boundary constraints
    A_ub_list = []
    b_ub_list = []
    
    # f(k) <= 10 for each k
    for i in range(m):
        row = np.zeros(m)
        row[i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(10.0)
    
    # f(k) >= -10 => -f(k) <= 10
    for i in range(m):
        row = np.zeros(m)
        row[i] = -1.0
        A_ub_list.append(row)
        b_ub_list.append(10.0)
    
    # Add initial coarse M(x) <= 1 constraints
    for x_idx in active_x_idx:
        A_ub_list.append(A_full[x_idx, :])
        b_ub_list.append(1.0)
    
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    # Solve LP iteratively with cutting plane
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Solve the LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs',
                          options={'time_limit': 30, 'presolve': True})
        except Exception:
            # If solve fails, try with more aggressive settings
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs',
                              options={'time_limit': 20})
            except Exception:
                break
        
        if not res.success:
            # Try to still get a feasible point
            if res.x is not None:
                f_vals = res.x
            else:
                break
        else:
            f_vals = res.x
        
        # Clip values to [-10, 10]
        f_vals = np.clip(f_vals, -10, 10)
        
        # Compute objective
        obj_val = -np.sum(f_vals * log_k_over_k)
        
        # Check if this is better than current best
        if obj_val > best_objective:
            best_objective = obj_val
            best_result = f_vals.copy()
        
        # Check for violated constraints: M(x) > 1.0001
        # M(x) = sum_k f(k) * floor(x/k)
        M_vals = A_full @ f_vals
        
        # Find violated x values
        violated = np.where(M_vals > 1.0001)[0]
        
        if len(violated) == 0:
            # No violations, we're done
            break
        
        # Add violated constraints (limit number per iteration to avoid explosion)
        # Add up to 20 new constraints per iteration
        new_constraints = violated[:20]
        for x_idx in new_constraints:
            if x_idx not in active_x_idx:
                active_x_idx.add(x_idx)
                A_ub_list.append(A_full[x_idx, :])
                b_ub_list.append(1.0)
        
        # Update A_ub
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        # If we've added all constraints, we can stop
        if len(active_x_idx) >= max_x:
            break
    
    # Use best result found
    if best_result is not None:
        f_vals = best_result
    elif 'res' in locals() and res.x is not None:
        f_vals = np.clip(res.x, -10, 10)
    else:
        # Fallback: use a simple feasible solution
        # mu(1)=1, mu(2)=-1, mu(3)=-1, mu(6)=1, others 0
        f_vals = np.zeros(m)
        if m >= 1:
            f_vals[0] = 1.0  # k=1
        if m >= 2:
            f_vals[1] = -1.0  # k=2
        if m >= 3:
            f_vals[2] = -1.0  # k=3
        if m >= 6:
            f_vals[5] = 1.0  # k=6
    
    # Now apply the evaluator's adjustment:
    # S = sum_k f(k)/k over clipped values
    # f(1) is replaced by f(1) - S so that sum_k f(k)/k = 0
    S = np.sum(f_vals / domain.astype(np.float64))
    
    # Adjust f(1)
    f_vals_adjusted = f_vals.copy()
    f_vals_adjusted[0] = f_vals_adjusted[0] - S  # k=1 is index 0
    
    # Verify feasibility after adjustment
    M_vals_check = A_full @ f_vals_adjusted
    if np.any(M_vals_check > 1.0001):
        # Infeasible, use a known feasible fallback
        f_vals_final = np.zeros(m)
        if m >= 1:
            f_vals_final[0] = 1.0
        if m >= 2:
            f_vals_final[1] = -1.0
        if m >= 3:
            f_vals_final[2] = -1.0
        if m >= 6:
            f_vals_final[5] = 1.0
        S2 = np.sum(f_vals_final / domain.astype(np.float64))
        f_vals_final[0] = f_vals_final[0] - S2
    else:
        f_vals_final = f_vals_adjusted
    
    # Filter out zero values to reduce domain size (optional but cleaner)
    nonzero_mask = np.abs(f_vals_final) > 1e-10
    if np.sum(nonzero_mask) > 0:
        domain_final = domain[nonzero_mask]
        values_final = f_vals_final[nonzero_mask]
    else:
        # Must have at least one element
        domain_final = domain[:1]
        values_final = f_vals_final[:1]
    
    return domain_final.astype(np.int64), values_final.astype(np.float64)


# EVOLVE-BLOCK-END
