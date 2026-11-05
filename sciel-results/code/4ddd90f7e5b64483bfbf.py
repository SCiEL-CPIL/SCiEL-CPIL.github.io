import time
import numpy as np
from scipy.optimize import linprog, SLSQP

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for the 90s wall-clock limit
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    V = np.array(types, dtype=float)
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # 1. Solve the allocation LP directly
    #    Variables: x_{i,g} for i in 0..15, g in 0..1
    #    Objective: Maximize E[revenue] = E[v.x] - E[U]
    #    We eliminate payment variables using Myerson's payment identity.
    #    For a single additive bidder, the optimal payment for a given allocation x is:
    #    p_i = v_i.x_i - U_i, where U_i = max_j [v_i.x_j - U_j] (envelope condition)
    #    Actually, for this small problem, let's just solve the joint LP directly 
    #    with tighter tolerances and a high-precision polish.
    
    x_alloc = _solve_allocation_lp(probs, V, n_types, n_goods)
    
    if time.time() > deadline - 15.0:
        # Fallback if LP is too slow
        final_alloc, final_pay = _simple_safe_mechanism(types)
        return {
            'allocations': final_alloc,
            'payments': final_pay
        }
    
    # 2. Compute payments from allocation using the optimal utility LP
    final_alloc, final_pay = _compute_payments_from_allocation(x_alloc, probs, V, n_types)
    
    if time.time() > deadline - 10.0:
        final_alloc = np.clip(final_alloc, 0.0, 1.0)
        final_pay = np.maximum(final_pay, 0.0)
        final_alloc, final_pay = _polish_mechanism(final_alloc, final_pay, probs, V, n_types)
        return {
            'allocations': final_alloc,
            'payments': final_pay
        }
    
    # 3. High-precision polish pass using SLSQP with tighter tolerances
    #    Try to improve the solution within the feasible region
    final_alloc, final_pay = _high_precision_polish(
        final_alloc, final_pay, probs, V, n_types, deadline
    )
    
    # 4. Final safety check
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    final_pay = np.maximum(final_pay, 0.0)
    final_alloc, final_pay = _polish_mechanism(final_alloc, final_pay, probs, V, n_types)
    
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
            s = arr.sum()
            if s > 0:
                return arr / s
            return np.ones(n_types) / n_types

    # If it's a dict
    if isinstance(distribution_spec, dict):
        for key in ['probs', 'distribution', 'type_probs', 'probabilities', 'p']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        arr = np.maximum(arr, 0)
                        s = arr.sum()
                        if s > 0:
                            return arr / s
                elif isinstance(val, dict):
                    probs = np.zeros(n_types)
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                    probs = np.maximum(probs, 0)
                    s = probs.sum()
                    if s > 0:
                        return probs / s
        
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
            s = probs.sum()
            if s > 0:
                return probs / s
    
    # Check support dict
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    s = arr.sum()
                    if s > 0:
                        return arr / s

    # Fallback: Uniform
    return np.ones(n_types) / n_types

def _solve_allocation_lp(probs, V, n_types, n_goods):
    """
    Solve the allocation LP with a utility variable approach.
    
    For a single additive bidder, the mechanism design problem can be written as:
    Maximize sum_i p_i * p_i (revenue)
    Subject to:
    - IC: U_i >= U_j + (v_i - v_j) . x_j for all i, j
    - IR: U_i >= 0 for all i
    - p_i = v_i . x_i - U_i
    - 0 <= x_{i,g} <= 1
    
    We can eliminate p_i using the payment identity and solve for x and U.
    However, the objective is nonlinear in U. Let's use the joint LP instead
    but with a more efficient formulation.
    
    Actually, for this specific case (single bidder, additive, finite types),
    the optimal mechanism is characterized by the allocation being monotone
    in each good. We can solve the LP directly.
    """
    return _solve_joint_lp_optimized(probs, V, n_types, n_goods)

def _solve_joint_lp_optimized(probs, V, n_types, n_goods):
    """
    Solve the joint LP for x and p with optimized constraint generation.
    
    Variables:
    x_{i,g} for i in 0..n_types-1, g in 0..n_goods-1  (n_types * n_goods vars)
    p_i for i in 0..n_types-1  (n_types vars)
    
    Total variables: n_types * (n_goods + 1) = 16 * 3 = 48
    """
    def var_x(i, g):
        return i * n_goods + g
    
    def var_p(i):
        return n_types * n_goods + i
    
    n_vars = n_types * (n_goods + 1)
    
    # Objective: maximize sum_i probs[i] * p[i]
    # So minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints:
    # 1. IC: v_i . x_i - p_i >= v_i . x_j - p_j  for all i, j
    #    => -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    # 2. IR: v_i . x_i - p_i >= 0  =>  -v_i . x_i + p_i <= 0
    # 3. Bounds: 0 <= x_{i,g} <= 1, p_i >= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = V[i]
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
        vi = V[i]
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0.0, 1.0)] * (n_types * n_goods) + [(0.0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": True, 
                               "dual_feasibility_tolerance": 1e-14, 
                               "primal_feasibility_tolerance": 1e-14,
                               "scale": True})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
            return allocations
        else:
            return np.zeros((n_types, n_goods))
    except Exception:
        return np.zeros((n_types, n_goods))

def _compute_payments_from_allocation(allocations, probs, V, n_types):
    """
    Given an allocation x, compute optimal payments using the utility LP.
    This ensures that for the given x, we extract maximum revenue.
    
    Variables: U_i for i in 0..n_types-1
    Objective: minimize sum_i probs[i] * U_i
    Constraints:
    1. -U_i <= 0  (U_i >= 0)
    2. -U_i + U_j <= v_j . x_j - v_i . x_j  (IC in utility form)
    """
    c = probs.copy()
    
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
        vi = V[i]
        for j in range(n_types):
            if i == j:
                continue
            vj = V[j]
            xj = allocations[j]
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0.0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": True,
                               "dual_feasibility_tolerance": 1e-14,
                               "primal_feasibility_tolerance": 1e-14})
        
        if res.success:
            U = res.x
            U = np.maximum(U, 0.0)
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = V[i]
                xi = allocations[i]
                p_i = np.dot(vi, xi) - U[i]
                if p_i < 0:
                    p_i = 0.0
                payments[i] = p_i
            
            return allocations, payments
        else:
            return _simple_safe_mechanism(n_types)
            
    except Exception:
        return _simple_safe_mechanism(n_types)

def _high_precision_polish(allocations, payments, probs, V, n_types, deadline):
    """
    Use SLSQP to refine the solution with tighter tolerances.
    We optimize the payment vector for a fixed allocation, or jointly
    optimize both with tight constraints.
    
    For a single additive bidder, once the allocation is fixed, the optimal
    payment is determined by the utility LP. So we can use SLSQP to refine
    the allocation while ensuring IC/IR hold.
    
    Alternative: Use SLSQP to jointly optimize x and p with tight constraints.
    """
    elapsed = time.time() - (deadline - 80.0)
    remaining = 80.0 - elapsed
    if remaining < 5.0:
        return allocations, payments
    
    # Joint optimization with SLSQP
    n_goods = 2
    x0 = np.concatenate([allocations.flatten(), payments])
    n_vars = n_types * (n_goods + 1)
    
    def var_x(i, g):
        return i * n_goods + g
    
    def var_p(i):
        return n_types * n_goods + i
    
    # Objective: maximize sum_i probs[i] * p[i]
    def objective(x):
        p = x[n_types * n_goods:]
        return -np.dot(probs, p)
    
    # Inequality constraints: IC and IR
    # IC: v_i . x_i - p_i - (v_i . x_j - p_j) >= 0
    # IR: v_i . x_i - p_i >= 0
    
    constraints = []
    
    # IC constraints
    for i in range(n_types):
        vi = V[i]
        for j in range(n_types):
            if i == j:
                continue
            # v_i . x_i - p_i - v_i . x_j + p_j >= 0
            def ic_con(x, i=i, j=j, vi=vi):
                xi = np.array([x[var_x(i, 0)], x[var_x(i, 1)]])
                xj = np.array([x[var_x(j, 0)], x[var_x(j, 1)]])
                pi = x[var_p(i)]
                pj = x[var_p(j)]
                return np.dot(vi, xi) - pi - (np.dot(vi, xj) - pj)
            constraints.append({'type': 'ineq', 'fun': ic_con})
    
    # IR constraints
    for i in range(n_types):
        vi = V[i]
        def ir_con(x, i=i, vi=vi):
            xi = np.array([x[var_x(i, 0)], x[var_x(i, 1)]])
            pi = x[var_p(i)]
            return np.dot(vi, xi) - pi
        constraints.append({'type': 'ineq', 'fun': ir_con})
    
    bounds = [(0.0, 1.0)] * (n_types * n_goods) + [(0.0, None)] * n_types
    
    try:
        res = SLSQP(x0, objective, bounds=bounds, constraints=constraints,
                     tol=1e-14, maxiter=100, disp=False)
        
        if res.success and res.fun < objective(x0):
            x_opt = res.x
            new_alloc = np.zeros((n_types, n_goods))
            new_pay = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    new_alloc[i, g] = x_opt[var_x(i, g)]
                new_pay[i] = x_opt[var_p(i)]
            
            # Verify and return if better
            revenue_old = np.dot(probs, payments)
            revenue_new = np.dot(probs, new_pay)
            if revenue_new > revenue_old - 1e-12:
                return new_alloc, new_pay
            else:
                return allocations, payments
        else:
            return allocations, payments
            
    except Exception:
        return allocations, payments

def _polish_mechanism(allocations, payments, probs, V, n_types):
    """
    Verify IC/IR constraints and repair violations with tight tolerance.
    """
    tol = 1e-10
    
    # Check IR: v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = V[i]
        xi = allocations[i]
        utility = np.dot(vi, xi) - payments[i]
        if utility < -tol:
            payments[i] = np.dot(vi, xi)
    
    # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
    # We iterate multiple times to handle cascading violations
    for iteration in range(10):
        changed = False
        for i in range(n_types):
            vi = V[i]
            for j in range(n_types):
                if i == j:
                    continue
                xi = allocations[i]
                xj = allocations[j]
                lhs = np.dot(vi, xi) - payments[i]
                rhs = np.dot(vi, xj) - payments[j]
                if lhs < rhs - tol:
                    # Increase payments[j] to fix IC violation
                    needed = rhs - lhs + tol
                    payments[j] += needed
                    changed = True
        
        if not changed:
            break
    
    # Re-check and clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Final strict check
    for i in range(n_types):
        vi = V[i]
        xi = allocations[i]
        if np.dot(vi, xi) - payments[i] < -1e-8:
            payments[i] = np.dot(vi, xi)
    
    for i in range(n_types):
        vi = V[i]
        for j in range(n_types):
            if i == j:
                continue
            xi = allocations[i]
            xj = allocations[j]
            lhs = np.dot(vi, xi) - payments[i]
            rhs = np.dot(vi, xj) - payments[j]
            if lhs < rhs - 1e-8:
                payments[j] += (rhs - lhs + 1e-8)
    
    return allocations, payments

def _simple_safe_mechanism(n_types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    """
    return np.zeros((n_types, 2)), np.zeros(n_types)
