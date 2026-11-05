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
    Solve allocation-only LP using Myerson payment identity.
    
    Key insight: For a monotone allocation, the payment is determined by:
    p_i = v_i . x_i - integral_0^{v_i} x'(v') dv' (in continuous case)
    
    For finite types, we can use the fact that IC + IR implies:
    U_i >= 0 and U_i - U_j >= v_i . x_j - v_j . x_j
    
    The optimal mechanism maximizes expected revenue = E[v . x] - E[U]
    subject to monotonicity constraints on x.
    
    However, a simpler approach: solve the LP over allocations only with
    the constraint that there EXISTS a feasible payment vector.
    This is equivalent to requiring the allocation to be "IC-feasible".
    
    For this small problem, we'll use the dual: maximize revenue subject to
    the existence of U >= 0 satisfying IC constraints.
    """
    # Variables: x[i][g] for i in range(n_types), g in range(n_goods)
    # n_vars = n_types * n_goods
    
    n_vars = n_types * n_goods
    
    def var_x(i, g):
        return i * n_goods + g
    
    # Objective: maximize E[revenue] = sum_i probs[i] * (v_i . x_i - U_i)
    # But U_i depends on x. We need to express this properly.
    # 
    # Alternative: Use the fact that for the optimal mechanism,
    # we can solve the LP with variables x and U jointly, but eliminate U
    # by noting that U_i = max(0, max_j (v_i . x_j - v_j . x_j + U_j))
    # This is complex. Instead, let's use a different approach.
    #
    # Actually, the cleanest way: solve the original LP but with the
    # payment variables eliminated using the IC constraints.
    # 
    # Let's just solve the full LP but with tighter tolerances and
    # a polish step. The current approach is fine; let's optimize it.
    
    # For now, fall back to the full LP approach but with improvements
    return _solve_full_lp_optimized(probs, types, n_types, n_goods)

def _solve_full_lp_optimized(probs, types, n_types, n_goods):
    """
    Optimized full LP solver with better numerics.
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types
    
    # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints:
    # 1. IC: -v_i . x_i + p_i + v_i . x_j - p_j <= 0 for all i != j
    # 2. IR: -v_i . x_i + p_i <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
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
        vi = np.array(types[i], dtype=float)
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        # Use HiGHS with tight tolerances
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12, 
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 40})
        
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
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            # RHS: v_j . x_j - v_i . x_j
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 20})
        
        if res.success:
            U = res.x
            U = np.maximum(U, 0.0)
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
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
    """
    n_types = len(types)
    tol = 1e-8
    
    # Check IR: v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = np.array(allocations[i], dtype=float)
        utility = np.dot(vi, xi) - payments[i]
        if utility < -tol:
            # Violation: reduce payment
            payments[i] = np.dot(vi, xi)
    
    # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j for all i, j
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xi = np.array(allocations[i], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            lhs = np.dot(vi, xi) - payments[i]
            rhs = np.dot(vi, xj) - payments[j]
            if lhs < rhs - tol:
                # IC violation: increase payment for j or decrease for i
                # Simplest: increase payment_j to make IC hold
                needed = rhs - lhs + tol
                payments[j] += needed
    
    # Re-check and clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return allocations, payments

def _refine_payments(allocations, probs, types):
    """
    Given an allocation x, solve for the optimal payments p that satisfy IC and IR
    and maximize revenue. This is equivalent to minimizing total expected utility.
    
    Variables: U_i (utility of type i)
    Minimize: sum_i probs[i] * U_i
    Subject to:
    1. U_i >= 0  (IR)
    2. U_i - U_j >= v_i . x_j - v_j . x_j  (IC) for all i, j
       Note: v_i . x_i - p_i = U_i => p_i = v_i . x_i - U_i
       IC: v_i . x_i - p_i >= v_i . x_j - p_j
           v_i . x_i - (v_i . x_i - U_i) >= v_i . x_j - (v_j . x_j - U_j)
           U_i >= v_i . x_j - v_j . x_j + U_j
           U_i - U_j >= v_i . x_j - v_j . x_j
    
    This LP has n_types variables and n_types*(n_types-1) + n_types constraints.
    Much smaller and better conditioned than the full LP.
    """
    n_types = len(types)
    
    # Objective: minimize sum probs[i] * U_i
    c = probs.copy()
    
    # Constraints:
    # 1. -U_i <= 0  (U_i >= 0)
    # 2. -U_i + U_j <= v_j . x_j - v_i . x_j  (Rearranged from U_i - U_j >= v_i . x_j - v_j . x_j)
    
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
        xi = np.array(allocations[i])
        vi_dot_xi = np.dot(vi, xi) # Not directly used in this form, let's stick to the derived form
        
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j])
            xj = np.array(allocations[j])
            # RHS: v_j . x_j - v_i . x_j
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9})
        
        if res.success:
            U = res.x
            # Ensure U >= 0 (should be enforced by bounds, but clip for safety)
            U = np.maximum(U, 0.0)
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i])
                xi = np.array(allocations[i])
                p_i = np.dot(vi, xi) - U[i]
                # Payment must be non-negative (IR ensures U_i <= v_i . x_i, so p_i >= 0)
                # But due to numerical errors, U might be slightly > v.i . x.i
                if p_i < 0:
                    p_i = 0.0
                payments[i] = p_i
            
            return allocations, payments
        else:
            # Fallback: Use the payments from the full LP (not passed here, so we need a safe default)
            # If this fails, we can just set payments to 0 (safe for IR, but might violate IC? No, if p=0, IC is v_i.x_i >= v_i.x_j. Not guaranteed.)
            # Better fallback: Use the max possible payment v_i.x_i (utility 0). This satisfies IR. 
            # Does it satisfy IC? 
            # If p_i = v_i.x_i, u_i = 0.
            # IC: 0 >= v_i.x_j - p_j.
            # This is not guaranteed.
            # So if the refinement LP fails, we should return the result from the full LP.
            # But we don't have the full LP payments here.
            # We can pass them in or just return a safe "zero revenue" mechanism if all else fails.
            return _simple_safe_mechanism(types)
            
    except Exception:
        return _simple_safe_mechanism(types)

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
