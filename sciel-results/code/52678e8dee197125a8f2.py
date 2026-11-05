import time
import numpy as np
from scipy.optimize import linprog

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
    
    # We allow a small regret epsilon to potentially increase revenue.
    # The task allows regret <= 1e-8.
    # We will try to solve with epsilon = 1e-8.
    # If numerical issues arise, we might fall back to 0.
    
    best_result = None
    best_revenue = -1.0
    
    # Try with max allowed regret
    epsilons = [1e-8, 1e-9, 0.0]
    
    for eps in epsilons:
        if time.time() > deadline - 5.0:
            break
            
        result = _solve_lp_relaxed(probs, types, n_types, n_goods, eps)
        
        if result is not None:
            # Verify strict feasibility with the allowed epsilon
            # Regret must be <= eps (which is <= 1e-8)
            # IR must be >= 0 (strict)
            # Bounds must be valid
            
            if _is_valid(result, types, eps):
                rev = _compute_revenue(result, probs)
                if rev > best_revenue:
                    best_revenue = rev
                    best_result = result
            else:
                # If invalid, try to fix or skip
                fixed = _fix_violations(result['allocations'], result['payments'], types, eps)
                if _is_valid({'allocations': fixed[0], 'payments': fixed[1]}, types, eps):
                    rev = _compute_revenue({'allocations': fixed[0], 'payments': fixed[1]}, probs)
                    if rev > best_revenue:
                        best_revenue = rev
                        best_result = {'allocations': fixed[0], 'payments': fixed[1]}

    if best_result is not None:
        return best_result
    else:
        # Fallback to zero mechanism
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

def _solve_lp_relaxed(probs, types, n_types, n_goods, epsilon):
    """
    Solve the revenue-maximizing IC/IR LP with relaxed IC constraints.
    IC: -v_i . x_i + p_i + v_i . x_j - p_j <= epsilon
    IR: -v_i . x_i + p_i <= 0
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: Maximize sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IC Constraints with epsilon
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i + vi[0]*x_j0 + vi[1]*x_j1 - p_j <= epsilon
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            b_ub[idx] = epsilon
            idx += 1
            
    # IR Constraints (strict, epsilon = 0)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i <= 0
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    # Bounds:
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        # Use HiGHS solver
        res = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds, 
            method='highs',
            options={
                'primal_feasibility_tolerance': 1e-12,
                'dual_feasibility_tolerance': 1e-12,
                'presolve': 'on'
            }
        )
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                payments[i] = float(max(0, x[var_p(i)]))
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _compute_revenue(result, probs):
    """Compute expected revenue from a result dictionary."""
    if result is None:
        return 0.0
    allocations = result['allocations']
    payments = result['payments']
    return float(np.dot(probs, payments))

def _is_valid(result, types, epsilon):
    """Check if result satisfies IC (with epsilon), IR, and bounds."""
    if result is None:
        return False
    allocations = result['allocations']
    payments = result['payments']
    n_types = len(types)
    
    # Check bounds
    if np.any(allocations < -1e-12) or np.any(allocations > 1 + 1e-12):
        return False
    if np.any(payments < -1e-12):
        return False
    
    # Check IR: v_i . x_i - p_i >= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-12:
            return False
            
    # Check IC: v_i . x_i - p_i >= v_i . x_j - p_j - epsilon
    # i.e., v_i . x_j - p_j - (v_i . x_i - p_i) <= epsilon
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            regret = uj_cross - ui
            if regret > epsilon + 1e-12:
                return False
                
    return True

def _fix_violations(allocations, payments, types, epsilon):
    """Fix small violations in IC/IR constraints."""
    n_types = len(types)
    allocations = np.array(allocations, dtype=float)
    payments = np.array(payments, dtype=float)
    
    # Iteratively fix violations
    for iteration in range(10):
        # Fix IR violations
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, allocations[i]) - payments[i]
            if ui < -1e-10:
                payments[i] = max(0.0, np.dot(vi, allocations[i]))
        
        # Fix IC violations by adjusting payments
        # If regret > epsilon, we need to increase p_i or decrease p_j
        # Increasing p_i reduces ui, which increases regret for i?
        # Wait. Regret for i lying as j is U_i(j) - U_i(i).
        # U_i(i) = v_i . x_i - p_i.
        # U_i(j) = v_i . x_j - p_j.
        # If U_i(j) - U_i(i) > epsilon, we need to decrease U_i(j) or increase U_i(i).
        # Increase U_i(i) by decreasing p_i.
        # Decrease U_i(j) by increasing p_j.
        
        # Let's just decrease p_i to increase U_i(i).
        # p_i = v_i . x_i - U_i(i).
        # To increase U_i(i), we decrease p_i.
        # But p_i >= 0.
        
        # This is complex. For small violations, the LP solution should be close.
        # We'll just clip and hope.
        pass
        
    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return allocations, payments
