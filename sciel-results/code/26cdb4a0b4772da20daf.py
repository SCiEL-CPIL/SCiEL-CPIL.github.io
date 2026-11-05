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
    deadline = start_time + 82.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    best_result = None
    best_revenue = -1.0
    
    epsilons = [1e-8, 0.0]
    
    for eps in epsilons:
        if time.time() > deadline - 5.0:
            break
        
        result = _solve_lp_relaxed(probs, types, n_types, n_goods, eps)
        
        if result is not None:
            if _is_valid(result, types, eps):
                rev = _compute_revenue(result, probs)
                if rev > best_revenue:
                    best_revenue = rev
                    best_result = result
            else:
                fixed = _fix_violations(result['allocations'], result['payments'], types, eps)
                fixed_result = {'allocations': fixed[0], 'payments': fixed[1]}
                if _is_valid(fixed_result, types, eps):
                    rev = _compute_revenue(fixed_result, probs)
                    if rev > best_revenue:
                        best_revenue = rev
                        best_result = fixed_result

    if best_result is not None:
        return best_result
    else:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

def _parse_distribution(distribution_spec, support, types, n_types):
    uniform = np.ones(n_types) / n_types
    
    if distribution_spec is None:
        return uniform
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            elif f"({t[0]}, {t[1]})" in distribution_spec:
                probs[i] = distribution_spec[f"({t[0]}, {t[1]})"]
                found_any = True
            elif f"{t[0]},{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]},{t[1]}"]
                found_any = True
            elif f"{t[0]}{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]}{t[1]}"]
                found_any = True
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
            return uniform
        return uniform
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        return uniform
    
    if isinstance(distribution_spec, str):
        if distribution_spec.lower() in ['uniform', 'unif']:
            return uniform
        try:
            import json
            spec = json.loads(distribution_spec)
            return _parse_distribution(spec, support, types, n_types)
        except:
            pass
        return uniform
    
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
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
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
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            b_ub[idx] = epsilon
            idx += 1
            
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        res = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds, 
            method='highs',
            options={
                'primal_feasibility_tolerance': 1e-14,
                'dual_feasibility_tolerance': 1e-14,
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
    if result is None:
        return 0.0
    return float(np.dot(probs, result['payments']))

def _is_valid(result, types, epsilon):
    if result is None:
        return False
    allocations = result['allocations']
    payments = result['payments']
    n_types = len(types)
    
    if np.any(allocations < -1e-12) or np.any(allocations > 1 + 1e-12):
        return False
    if np.any(payments < -1e-12):
        return False
    
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-12:
            return False
            
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
    n_types = len(types)
    allocations = np.array(allocations, dtype=float)
    payments = np.array(payments, dtype=float)
    
    for iteration in range(20):
        changed = False
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, allocations[i]) - payments[i]
            if ui < -1e-10:
                payments[i] = max(0.0, np.dot(vi, allocations[i]))
                changed = True
        
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, allocations[i]) - payments[i]
            for j in range(n_types):
                if i == j:
                    continue
                uj_cross = np.dot(vi, allocations[j]) - payments[j]
                regret = uj_cross - ui
                if regret > epsilon + 1e-12:
                    new_p_i = max(0.0, np.dot(vi, allocations[i]) - (uj_cross - epsilon))
                    if new_p_i < payments[i]:
                        payments[i] = new_p_i
                        changed = True
        
        if not changed:
            break
    
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return allocations, payments
