import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # 1. Solve allocation-only LP using Myerson payment identity
    #    This eliminates payment variables and ensures IC by construction
    x_alloc = _solve_allocation_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 5.0:
        # Fallback if LP is too slow
        return _simple_safe_mechanism(types)
    
    # 2. Compute payments from allocation using the optimal utility LP
    final_alloc, final_pay = _compute_payments_from_allocation(x_alloc, probs, types)
    
    # 3. Final safety check and clipping
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    final_pay = np.maximum(final_pay, 0.0)
    
    # 4. Polish pass: verify constraints and repair if needed
    final_alloc, final_pay = _polish_mechanism(final_alloc, final_pay, probs, types)
    
    return {
        'allocations': final_alloc,
        'payments': final_pay
    }

def _parse_distribution(distribution_spec, support, types, n_types):
    """Robustly parse distribution spec to get type probabilities."""
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    # If it's a direct array of correct size
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
            return np.ones(n_types) / n_types

    # If it's a dict
    if isinstance(distribution_spec, dict):
        # Look for common keys
        for key in ['probs', 'distribution', 'type_probs', 'probabilities', 'p']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        arr = np.maximum(arr, 0)
                        if arr.sum() > 0:
                            return arr / arr.sum()
                elif isinstance(val, dict):
                    # Map tuples to probs
                    probs = np.zeros(n_types)
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()
        
        # Check if the dict itself maps types to probs
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
                found_any = True
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
    
    # Check support dict
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    # Fallback: Uniform
    return np.ones(n_types) / n_types

def _solve_allocation_lp(probs, types, n_types, n_goods):
    """
    Solve the allocation LP directly.
    Objective: Maximize Expected Revenue = E[v.x] - E[U]
    Using the fact that for any allocation x, the minimal expected utility
    (and thus maximal revenue) is determined by the LP over U.
    
    However, for a single additive bidder, the optimal mechanism is characterized
    by monotonicity. But solving the joint LP (x, p) is robust and small enough.
    Let's stick to the robust joint LP but optimize the construction.
    """
    return _solve_joint_lp(probs, types, n_types, n_goods)

def _solve_joint_lp(probs, types, n_types, n_goods):
    """
    Solve the joint LP for x and p.
    Variables:
    x_{i,g} for i in 0..15, g in 0..1
    p_i for i in 0..15
    
    Objective: Maximize sum_i probs[i] * p_i
    Subject to:
    1. IC: v_i . x_i - p_i >= v_i . x_j - p_j  =>  -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    2. IR: v_i . x_i - p_i >= 0  =>  -v_i . x_i + p_i <= 0
    3. Bounds: 0 <= x_{i,g} <= 1, p_i >= 0
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types
    
    # Objective: minimize -sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j:
                continue
            # -vi.x_i + p_i + vi.x_j - p_j <= 0
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            idx += 1
    
    for i in range(n_types):
        vi = np.array(types[i])
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12, 
                               "primal_feasibility_tolerance": 1e-12})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                payments[i] = x[var_p(i)]
            return allocations
        else:
            return np.zeros((n_types, n_goods))
    except Exception:
        return np.zeros((n_types, n_goods))

def _compute_payments_from_allocation(allocations, probs, types):
    """
    Given an allocation x, compute optimal payments using the utility LP.
    This ensures that for the given x, we extract maximum revenue.
    """
    n_types = len(types)
    
    # Objective: minimize sum probs[i] * U_i
    c = probs.copy()
    
    # Constraints:
    # 1. -U_i <= 0  (U_i >= 0)
    # 2. -U_i + U_j <= v_j . x_j - v_i . x_j
    
    n_vars = n_types
    n_ineq = n_types + n_types * (n_types - 1)
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IR: -U_i <= 0
    for i in range(n_types):
        A_ub[idx, i] = -1.0
        b_ub[idx] = 0.0
        idx += 1
    
    # IC: -U_i + U_j <= v_j . x_j - v_i . x_j
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j])
            xj = np.array(allocations[j])
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12})
        
        if res.success:
            U = res.x
            U = np.maximum(U, 0.0)
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i])
                xi = np.array(allocations[i])
                p_i = np.dot(vi, xi) - U[i]
                if p_i < 0:
                    p_i = 0.0
                payments[i] = p_i
            
            return allocations, payments
        else:
            return _simple_safe_mechanism(types)
            
    except Exception:
        return _simple_safe_mechanism(types)

def _polish_mechanism(allocations, payments, probs, types):
    """
    Verify IC/IR constraints and repair violations.
    This is a critical step to ensure the output is strictly valid
    within the required tolerance (1e-8).
    """
    n_types = len(types)
    tol = 1e-9  # Use tighter internal tolerance for safety
    
    # Check IR: v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        utility = np.dot(vi, xi) - payments[i]
        if utility < -tol:
            # Violation: reduce payment to make utility 0
            payments[i] = np.dot(vi, xi)
    
    # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j])
            xi = np.array(allocations[i])
            xj = np.array(allocations[j])
            lhs = np.dot(vi, xi) - payments[i]
            rhs = np.dot(vi, xj) - payments[j]
            if lhs < rhs - tol:
                # IC violation: increase payment_j to make IC hold
                # We want lhs >= rhs.
                # lhs - rhs < -tol
                # We can increase payments[j] which decreases rhs.
                # New rhs = rhs - delta
                # We need lhs >= rhs - delta
                # delta >= rhs - lhs
                needed = rhs - lhs + tol
                payments[j] += needed
    
    # Re-check and clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Final strict check and repair
    for i in range(n_types):
        vi = np.array(types[i])
        xi = np.array(allocations[i])
        if np.dot(vi, xi) - payments[i] < -1e-8:
             payments[i] = np.dot(vi, xi)
    
    for i in range(n_types):
        vi = np.array(types[i])
        for j in range(n_types):
            if i == j: continue
            vj = np.array(types[j])
            xi = np.array(allocations[i])
            xj = np.array(allocations[j])
            lhs = np.dot(vi, xi) - payments[i]
            rhs = np.dot(vi, xj) - payments[j]
            if lhs < rhs - 1e-8:
                 payments[j] += (rhs - lhs + 1e-8)
    
    return allocations, payments

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
