import time
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize as sp_minimize

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible and individually rational auction mechanism.
    
    Args:
        distribution_spec: Specification of the probability distribution over types.
        support: The support of the types (usually {(0,0), ..., (3,3)}).
        constraints: Additional constraints (not explicitly detailed, assumed standard).
        
    Returns:
        A dictionary with 'allocations' (16, 2) and 'payments' (16,).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 10.0:
        # Fallback to zero mechanism if running out of time
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    if result is not None:
        # Try to polish with SLSQP for better precision and potential improvement
        polished = _polish_mechanism(result, probs, types, deadline)
        if polished is not None:
            return polished
        return result
    else:
        # Fallback to zero mechanism if LP fails
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

def _parse_distribution(distribution_spec, support, types, n_types):
    """
    Parse the distribution spec into an array of probabilities for the 16 types.
    Order: (0,0), (0,1), ..., (0,3), (1,0), ..., (3,3).
    """
    uniform = np.ones(n_types) / n_types
    
    if distribution_spec is None:
        return uniform
    
    # If it's a dict mapping types to probs
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            # Try tuple key
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            # Try string key "(v1, v2)"
            elif f"({t[0]}, {t[1]})" in distribution_spec:
                probs[i] = distribution_spec[f"({t[0]}, {t[1]})"]
                found_any = True
            # Try string key "v1,v2"
            elif f"{t[0]},{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]},{t[1]}"]
                found_any = True
            # Try flattened string key "v1v2"
            elif f"{t[0]}{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]}{t[1]}"]
                found_any = True
        
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
            return uniform
        return uniform
    
    # If it's a list or array
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        
        # Maybe it's a 4x4 matrix?
        if len(arr) == 16:
             # Already handled above
             pass
             
        # Maybe it's product of marginals? E.g., [p0, p1, p2, p3] for each good?
        # This is a guess. If length is 4, assume independent marginals.
        if len(arr) == 4:
            # arr[i] is prob for value i?
            # This is ambiguous. Let's stick to uniform if not 16.
            pass
        
        return uniform
    
    # If it's a string, might be a name
    if isinstance(distribution_spec, str):
        if distribution_spec.lower() in ['uniform', 'unif']:
            return uniform
        # Try to parse JSON?
        try:
            import json
            spec = json.loads(distribution_spec)
            return _parse_distribution(spec, support, types, n_types)
        except:
            pass
        return uniform

    # Fallback: Check support
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    return uniform

def _solve_lp(probs, types, n_types, n_goods):
    """
    Solve the revenue-maximizing IC/IR LP.
    Variables: x[0..2*n_types-1] for allocations, x[2*n_types..3*n_types-1] for payments.
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we use negative.
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints:
    # 1. IC: for all i != j, v_i . x_i - p_i >= v_i . x_j - p_j
    #    => -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    # 2. IR: for all i, v_i . x_i - p_i >= 0
    #    => -v_i . x_i + p_i <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IC Constraints
    for i in range(n_types):
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = types[j]
            # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i + vi[0]*x_j0 + vi[1]*x_j1 - p_j <= 0
            A_ub[idx, var_x(i, 0)] = -vi_vec[0]
            A_ub[idx, var_x(i, 1)] = -vi_vec[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi_vec[0]
            A_ub[idx, var_x(j, 1)] = vi_vec[1]
            A_ub[idx, var_p(j)] = -1.0
            b_ub[idx] = 0.0
            idx += 1
            
    # IR Constraints
    for i in range(n_types):
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i <= 0
        A_ub[idx, var_x(i, 0)] = -vi_vec[0]
        A_ub[idx, var_x(i, 1)] = -vi_vec[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    # Bounds:
    # Allocations x[i,g] in [0, 1]
    # Payments p[i] in [0, infinity)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        # Use HiGHS solver with tight tolerances for better precision
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_objective_bound_tolerance': 1e-12})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    # Clip to ensure feasibility
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                # Ensure non-negative payment
                payments[i] = float(max(0, x[var_p(i)]))
            
            # Final verification and tiny fix for numerical drift
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _verify_and_fix(allocations, payments, types):
    """
    Verify IC/IR constraints and fix small numerical violations.
    """
    n_types = len(types)
    
    # Check IR and fix
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-10:
            # Payment is too high, reduce it to satisfy IR
            payments[i] = max(0, np.dot(vi, allocations[i]))
            
    # Check IC and fix
    # If IC is violated, try to fix by adjusting payments or allocations
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - 1e-8:
                # Violation found - increase payment for type i to reduce its utility
                needed_increase = uj_cross - ui + 1e-9
                payments[i] += needed_increase
                # Ensure IR is still satisfied
                if payments[i] > np.dot(vi, allocations[i]):
                    # Scale down allocation for type i
                    if np.dot(vi, allocations[i]) > 1e-12:
                        scale = (uj_cross + 1e-9) / np.dot(vi, allocations[i])
                        if scale < 1:
                            allocations[i] *= scale
                    else:
                        allocations[i] = 0
                    payments[i] = max(0, np.dot(vi, allocations[i]) - max(0, uj_cross - 1e-9))

    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # One more pass to ensure IR feasibility
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-8:
            payments[i] = max(0, np.dot(vi, allocations[i]))
    
    return allocations, payments

def _polish_mechanism(result, probs, types, deadline):
    """
    Polish the LP solution using SLSQP to potentially improve revenue while maintaining IC/IR.
    """
    if time.time() > deadline - 5.0:
        return None
    
    allocations = result['allocations'].copy()
    payments = result['payments'].copy()
    n_types = len(types)
    n_goods = 2
    
    # Flatten variables: [alloc_00, alloc_01, ..., alloc_31, pay_0, ..., pay_3]
    x0 = np.concatenate([allocations.flatten(), payments])
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    def objective(x):
        alloc = x[:n_types * n_goods].reshape(n_types, n_goods)
        pay = x[n_types * n_goods:]
        return -np.sum(probs * pay)
    
    # Constraints: IC and IR
    def constraints(x):
        alloc = x[:n_types * n_goods].reshape(n_types, n_goods)
        pay = x[n_types * n_goods:]
        
        cons = []
        
        # IC: u_i >= u_i|_j for all i != j
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, alloc[i]) - pay[i]
            for j in range(n_types):
                if i == j:
                    continue
                uj_cross = np.dot(vi, alloc[j]) - pay[j]
                cons.append(ui - uj_cross)
        
        # IR: u_i >= 0
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, alloc[i]) - pay[i]
            cons.append(ui)
        
        return cons
    
    # Bounds: allocations in [0,1], payments >= 0
    bounds = []
    for i in range(n_types * n_goods):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    try:
        res = sp_minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints={'type': 'ineq', 'fun': constraints},
                         options={'ftol': 1e-12, 'maxiter': 100})
        
        if time.time() > deadline - 3.0:
            return None
        
        if res.success:
            new_alloc = res.x[:n_types * n_goods].reshape(n_types, n_goods)
            new_pay = res.x[n_types * n_goods:]
            
            # Verify feasibility
            new_alloc, new_pay = _verify_and_fix(new_alloc, new_pay, types)
            
            # Check if improved
            old_rev = np.sum(probs * payments)
            new_rev = np.sum(probs * new_pay)
            
            if new_rev > old_rev - 1e-10:
                return {'allocations': new_alloc, 'payments': new_pay}
            else:
                return result
        else:
            return result
    except Exception:
        return result
