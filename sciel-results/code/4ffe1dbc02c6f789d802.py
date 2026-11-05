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
    
    # Solve the refined LP
    result = _solve_refined_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 5.0:
        # Fallback if LP is too slow
        return _simple_safe_mechanism(types)
    
    if result is None:
        return _simple_safe_mechanism(types)
    
    final_alloc, final_pay = result
    
    # Final safety check and clipping
    # Ensure allocations are strictly within [0,1]
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    # Payments should be non-negative by construction, but clip just in case
    final_pay = np.maximum(final_pay, 0.0)
    
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

def _solve_refined_lp(probs, types, n_types, n_goods):
    """
    Solve the mechanism design LP with variables x[i][g] and U[i].
    Includes explicit constraint p_i >= 0 which translates to U_i <= v_i . x_i.
    
    Variables:
    - x[i][g] for i in range(n_types), g in range(n_goods): allocation probabilities
    - U[i] for i in range(n_types): utility of type i
    
    Objective: maximize sum_i probs[i] * (v_i . x_i - U[i])
    This equals expected revenue.
    
    Constraints:
    1. IC: U_i - U_j >= v_i . x_j - v_j . x_j for all i, j
       Equivalent to: -U_i + U_j <= v_j . x_j - v_i . x_j
    2. IR: U_i >= 0
    3. Non-negative Payment: p_i = v_i . x_i - U_i >= 0  =>  U_i - v_i . x_i <= 0
    4. 0 <= x[i][g] <= 1
    """
    # Variable indexing
    def var_x(i, g):
        return i * n_goods + g

    def var_U(i):
        return n_types * n_goods + i

    n_vars = n_types * n_goods + n_types
    
    # Objective: minimize -sum_i probs[i] * (v_i . x_i - U[i])
    c = np.zeros(n_vars)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for g in range(n_goods):
            c[var_x(i, g)] = -probs[i] * vi[g]
        c[var_U(i)] = probs[i]  # minimize -(... - U_i) => coefficient of U_i is +probs[i]
    
    # Build inequality constraints A_ub @ var <= b_ub
    
    # 1. IC constraints: -U_i + U_j + (vi - vj) . x_j <= 0
    # 2. IR constraints: -U_i <= 0
    # 3. Non-neg Payment: U_i - vi . x_i <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_np = n_types
    n_ineq = n_ic + n_ir + n_np
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    
    # IC Constraints
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            # -U_i + U_j + (vi - vj) . x_j <= 0
            A_ub[idx, var_U(i)] = -1.0
            A_ub[idx, var_U(j)] = 1.0
            for g in range(n_goods):
                A_ub[idx, var_x(j, g)] = vi[g] - vj[g]
            idx += 1
    
    # IR Constraints
    for i in range(n_types):
        A_ub[idx, var_U(i)] = -1.0
        idx += 1
        
    # Non-negative Payment Constraints: U_i - sum_g vi[g] x_i[g] <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        A_ub[idx, var_U(i)] = 1.0
        for g in range(n_goods):
            A_ub[idx, var_x(i, g)] = -vi[g]
        idx += 1
    
    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 40})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]
            
            # Compute payments: p_i = v_i . x_i - U_i
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - utilities[i]
                # Due to the constraint U_i <= v_i . x_i, p_i should be >= 0.
                # Clip to 0 to handle tiny numerical negatives.
                payments[i] = max(0.0, p_i)
            
            return allocations, payments
        else:
            return None
    except Exception:
        return None

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
