"""Incentive-compatible finite auction design solver using LP + SLSQP polish."""
import time
import numpy as np
from scipy.optimize import linprog, minimize

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    lp_result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 3.0:
        if lp_result is not None:
            return lp_result
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    best_alloc = None
    best_pay = None
    best_rev = -np.inf
    
    if lp_result is not None:
        best_alloc = lp_result['allocations']
        best_pay = lp_result['payments']
        best_rev = _compute_revenue(best_alloc, best_pay, probs)
    
    # Try SLSQP polish from LP solution and perturbed variants
    if lp_result is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 5.0:
            slsqp_result = _slsqp_polish(probs, types, n_types, n_goods,
                                          best_alloc, best_pay, remaining_time)
            if slsqp_result is not None:
                slsqp_alloc, slsqp_pay = slsqp_result
                slsqp_rev = _compute_revenue(slsqp_alloc, slsqp_pay, probs)
                if slsqp_rev > best_rev:
                    best_alloc = slsqp_alloc
                    best_pay = slsqp_pay
                    best_rev = slsqp_rev
    
    if best_alloc is None:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    return {'allocations': best_alloc, 'payments': best_pay}

def _aggressive_fix(allocations, payments, types, probs):
    """More aggressive fixing of IC/IR violations."""
    n_types = len(types)
    
    # First fix IR by reducing payments
    for i in range(n_types):
        vi = types[i]
        max_pay = np.dot(vi, allocations[i])
        payments[i] = min(payments[i], max_pay)
    
    # Then fix IC by adjusting payments
    for iteration in range(10):
        violated = False
        for i in range(n_types):
            vi = types[i]
            ui = np.dot(vi, allocations[i]) - payments[i]
            for j in range(n_types):
                if i == j:
                    continue
                uj_cross = np.dot(vi, allocations[j]) - payments[j]
                if ui < uj_cross - 1e-10:
                    violated = True
                    # Increase payment for j to make uj_cross <= ui
                    needed_pay_j = np.dot(vi, allocations[j]) - ui
                    if needed_pay_j > payments[j]:
                        payments[j] = needed_pay_j
                    # Also ensure IR for j
                    vj = types[j]
                    max_pay_j = np.dot(vj, allocations[j])
                    payments[j] = min(payments[j], max_pay_j)
        
        if not violated:
            break
    
    payments = np.maximum(payments, 0)
    allocations = np.clip(allocations, 0, 1)
    
    return allocations, payments

def _parse_distribution(distribution_spec, support, types, n_types):
    """Parse distribution spec to get type probabilities."""
    # Try various formats
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, str):
        # Maybe a name like "uniform", "independent", etc.
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        # Try various keys
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
                    probs = np.ones(n_types) / n_types
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                        elif f"{t[0]},{t[1]}" in val:
                            probs[i] = val[f"{t[0]},{t[1]}"]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()
        
        # Maybe the dict maps type tuples to probs directly
        probs = np.zeros(n_types)
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
        if probs.sum() > 0:
            return probs / probs.sum()
    
    # Fallback: check support
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()
    
    return np.ones(n_types) / n_types

def _solve_lp(probs, types, n_types, n_goods):
    """Solve the revenue-maximizing IC/IR LP."""
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 2 * n_types + n_types
    
    # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # IC: for all i != j: v_i·x_i - p_i >= v_i·x_j - p_j
    # => -v_i·x_i + p_i + v_i·x_j - p_j <= 0
    # IR: for all i: v_i·x_i - p_i >= 0 => -v_i·x_i + p_i <= 0
    
    n_ic = n_types * n_types
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = types[i]
        for j in range(n_types):
            if i == j:
                idx += 1
                continue
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            idx += 1
    
    for i in range(n_types):
        vi = types[i]
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        idx += 1
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                payments[i] = float(max(0, x[var_p(i)]))
            
            # Post-process: verify and fix small violations
            allocations, payments = _verify_and_fix(allocations, payments, types, probs)
            
            # Final validity check
            if _check_validity(allocations, payments, types, probs):
                return {'allocations': allocations, 'payments': payments}
            else:
                # Try to fix more aggressively
                allocations, payments = _aggressive_fix(allocations, payments, types, probs)
                if _check_validity(allocations, payments, types, probs):
                    return {'allocations': allocations, 'payments': payments}
                return None
        else:
            return None
    except Exception:
        return None

def _compute_revenue(allocations, payments, probs):
    """Compute expected revenue."""
    return np.dot(probs, payments)

def _check_validity(allocations, payments, types, probs):
    """Check if mechanism is IC and IR with tolerance 1e-8."""
    n_types = len(types)
    tol = 1e-8
    
    # Check IR
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -tol:
            return False
    
    # Check IC
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - tol:
                return False
    
    # Check non-negative payments
    if np.any(payments < -tol):
        return False
    
    # Check allocation bounds
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    return True

def _slsqp_polish(probs, types, n_types, n_goods, init_alloc, init_pay, time_budget):
    """Use SLSQP to polish the LP solution, maximizing revenue subject to IC/IR."""
    
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    def unpack(x):
        alloc = np.zeros((n_types, n_goods))
        pay = np.zeros(n_types)
        for i in range(n_types):
            for g in range(n_goods):
                alloc[i, g] = x[var_x(i, g)]
            pay[i] = x[var_p(i)]
        return alloc, pay
    
    def objective(x):
        alloc, pay = unpack(x)
        return -np.dot(probs, pay)
    
    # Constraints: IC and IR as inequality constraints (g(x) >= 0 for SLSQP)
    # IC: v_i·x_i - p_i - (v_i·x_j - p_j) >= 0 for all i != j
    # IR: v_i·x_i - p_i >= 0 for all i
    
    cons = []
    
    # IC constraints
    for i in range(n_types):
        vi = types[i]
        for j in range(n_types):
            if i == j:
                continue
            def ic_con(x, i=i, j=j, vi=vi):
                alloc, pay = unpack(x)
                ui = np.dot(vi, alloc[i]) - pay[i]
                uj_cross = np.dot(vi, alloc[j]) - pay[j]
                return ui - uj_cross
            cons.append({'type': 'ineq', 'fun': ic_con})
    
    # IR constraints
    for i in range(n_types):
        vi = types[i]
        def ir_con(x, i=i, vi=vi):
            alloc, pay = unpack(x)
            return np.dot(vi, alloc[i]) - pay[i]
        cons.append({'type': 'ineq', 'fun': ir_con})
    
    # Bounds
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
        bounds.append((0, None))
    
    # Initial point from LP solution
    x0 = np.zeros(n_vars)
    for i in range(n_types):
        for g in range(n_goods):
            x0[var_x(i, g)] = init_alloc[i, g]
        x0[var_p(i)] = init_pay[i]
    
    # Try SLSQP from the LP solution
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12})
        
        if res.success or res.fun < 0:
            alloc, pay = unpack(res.x)
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            if _check_validity(alloc, pay, types, probs):
                return alloc, pay
    except Exception:
        pass
    
    # Try with small perturbations if time allows
    if time_budget > 15.0:
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if time.time() > time_budget + 2.0:
                break
            
            x0_pert = x0.copy()
            noise = rng.uniform(-0.05, 0.05, size=n_vars)
            x0_pert += noise
            x0_pert[:2*n_types] = np.clip(x0_pert[:2*n_types], 0, 1)
            x0_pert[2*n_types:] = np.maximum(x0_pert[2*n_types:], 0)
            
            try:
                res = minimize(objective, x0_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 150, 'ftol': 1e-12})
                
                if res.success or res.fun < 0:
                    alloc, pay = unpack(res.x)
                    alloc = np.clip(alloc, 0, 1)
                    pay = np.maximum(pay, 0)
                    
                    if _check_validity(alloc, pay, types, probs):
                        rev = _compute_revenue(alloc, pay, probs)
                        base_rev = _compute_revenue(init_alloc, init_pay, probs)
                        if rev > base_rev - 1e-10:
                            return alloc, pay
            except Exception:
                pass
    
    return None

def _verify_and_fix(allocations, payments, types, probs):
    """Verify IC/IR and fix small numerical violations."""
    n_types = len(types)
    n_goods = allocations.shape[1]
    
    # Check and fix IR
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-10:
            # Reduce payment to satisfy IR
            payments[i] = np.dot(vi, allocations[i])
    
    # Check IC and fix
    for i in range(n_types):
        vi = types[i]
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - 1e-10:
                # Reduce allocation to j or increase payment to j
                # Simple fix: increase payment for j
                payments[j] = max(payments[j], np.dot(vi, allocations[j]) - ui)
    
    # Ensure non-negative
    payments = np.maximum(payments, 0)
    allocations = np.clip(allocations, 0, 1)
    
    return allocations, payments
