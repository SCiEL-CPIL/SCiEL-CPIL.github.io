import time
import numpy as np
from scipy.optimize import linprog, minimize

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
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Solve the LP to get the optimal solution
    lp_result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 10.0:
        # Fallback to zero mechanism if running out of time
        if lp_result is not None:
            return lp_result
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    best_result = None
    best_revenue = -1.0

    if lp_result is not None:
        # Calculate revenue for LP result
        rev_lp = np.dot(probs, lp_result['payments'])
        best_result = lp_result
        best_revenue = rev_lp

        # Use LP solution as warm start for SLSQP refinement
        slsqp_result = _solve_slsqp(probs, types, n_types, n_goods, lp_result)
        if slsqp_result is not None:
            rev_slsqp = np.dot(probs, slsqp_result['payments'])
            if rev_slsqp > best_revenue:
                best_result = slsqp_result
                best_revenue = rev_slsqp

    if best_result is None:
        # LP failed, try SLSQP from scratch
        slsqp_result = _solve_slsqp(probs, types, n_types, n_goods, None)
        if slsqp_result is not None:
            best_result = slsqp_result
        else:
            # Fallback to zero mechanism
            best_result = {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
            
    return best_result

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
        # Use HiGHS solver, which is robust and fast for this size
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
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

def _solve_slsqp(probs, types, n_types, n_goods, warm_start=None):
    """
    Solve the revenue-maximizing IC/IR problem using SLSQP with explicit
    differentiable constraints. This avoids LP linearization artifacts and
    can achieve higher precision.
    
    Variables: z = [x_0,0, x_0,1, ..., x_15,0, x_15,1, p_0, ..., p_15]
    Total: 2*n_types + n_types = 3*n_types = 48 variables
    
    Objective: Minimize -sum_i probs[i] * p[i]  (maximize revenue)
    
    Constraints (all <= 0):
    1. IC: for all i != j: v_i . x_i - p_i - (v_i . x_j - p_j) <= 0
       i.e., utility_i(i) <= utility_i(j) is WRONG.
       Correct: utility_i(i) >= utility_i(j) for all i,j
       So: utility_i(j) - utility_i(i) <= 0
       
    2. IR: for all i: 0 - (v_i . x_i - p_i) <= 0
       i.e., -utility_i(i) <= 0
    """
    n_vars = 3 * n_types
    
    def var_x_idx(i, g):
        return 2 * i + g
    
    def var_p_idx(i):
        return 2 * n_types + i
    
    # Precompute value vectors
    V = np.array(types, dtype=float)  # shape (n_types, 2)
    
    # Objective: minimize -expected revenue
    def objective(z):
        payments = z[2*n_types:]
        return -np.dot(probs, payments)
    
    # Objective gradient
    def objective_grad(z):
        grad = np.zeros(n_vars)
        grad[2*n_types:] = -probs
        return grad
    
    # Constraints: each returns array of constraint values (all should be <= 0)
    
    # IC constraints: for each type i, for each other type j (j != i):
    #   v_i . x_j - p_j - (v_i . x_i - p_i) <= 0
    # This means: the cross-utility of reporting j when type is i should be 
    # <= the true utility of type i.
    
    # Let's build the constraint Jacobian and function
    
    # Number of IC constraints: n_types * (n_types - 1) = 16 * 15 = 240
    # Number of IR constraints: n_types = 16
    # Total: 256 constraints
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_constraints = n_ic + n_ir
    
    # Build constraint matrix structure
    # For IC constraint (i, j): v_i . x_j - p_j - v_i . x_i + p_i <= 0
    # Derivatives w.r.t. x_j: +v_i
    # Derivatives w.r.t. p_j: -1
    # Derivatives w.r.t. x_i: -v_i
    # Derivatives w.r.t. p_i: +1
    
    # For IR constraint i: -(v_i . x_i - p_i) <= 0
    # Derivatives w.r.t. x_i: -v_i
    # Derivatives w.r.t. p_i: +1
    
    # Precompute constraint indices for efficiency
    ic_idx = []
    for i in range(n_types):
        for j in range(n_types):
            if i != j:
                ic_idx.append((i, j))
    
    def constraints(z):
        """Return array of constraint values, all should be <= 0."""
        cvals = np.zeros(n_constraints)
        
        # IC constraints
        for k, (i, j) in enumerate(ic_idx):
            vi = V[i]
            # utility_i_as_j = vi . x_j - p_j
            # utility_i_true = vi . x_i - p_i
            # constraint: utility_i_as_j - utility_i_true <= 0
            u_cross = np.dot(vi, z[var_x_idx(j, 0):var_x_idx(j, 0)+2]) - z[var_p_idx(j)]
            u_true = np.dot(vi, z[var_x_idx(i, 0):var_x_idx(i, 0)+2]) - z[var_p_idx(i)]
            cvals[k] = u_cross - u_true
        
        # IR constraints
        for i in range(n_types):
            vi = V[i]
            u_true = np.dot(vi, z[var_x_idx(i, 0):var_x_idx(i, 0)+2]) - z[var_p_idx(i)]
            cvals[n_ic + i] = -u_true
        
        return cvals
    
    def constraints_jacobian(z):
        """Return sparse-like Jacobian as dense matrix (n_constraints x n_vars)."""
        J = np.zeros((n_constraints, n_vars))
        
        # IC constraints
        for k, (i, j) in enumerate(ic_idx):
            vi = V[i]
            # d/dx_j: +vi
            J[k, var_x_idx(j, 0)] = vi[0]
            J[k, var_x_idx(j, 1)] = vi[1]
            # d/dp_j: -1
            J[k, var_p_idx(j)] = -1.0
            # d/dx_i: -vi
            J[k, var_x_idx(i, 0)] = -vi[0]
            J[k, var_x_idx(i, 1)] = -vi[1]
            # d/dp_i: +1
            J[k, var_p_idx(i)] = 1.0
        
        # IR constraints
        for i in range(n_types):
            vi = V[i]
            row = n_ic + i
            # d/dx_i: -vi
            J[row, var_x_idx(i, 0)] = -vi[0]
            J[row, var_x_idx(i, 1)] = -vi[1]
            # d/dp_i: +1
            J[row, var_p_idx(i)] = 1.0
        
        return J
    
    # Bounds: x in [0,1], p in [0, inf)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Initial point
    if warm_start is not None:
        z0 = np.zeros(n_vars)
        for i in range(n_types):
            for g in range(n_goods):
                z0[var_x_idx(i, g)] = warm_start['allocations'][i, g]
            z0[var_p_idx(i)] = warm_start['payments'][i]
    else:
        # Start from a simple feasible point: zero allocations, zero payments
        z0 = np.zeros(n_vars)
    
    # Verify initial point feasibility
    c0 = constraints(z0)
    if np.any(c0 > 1e-6):
        # Try to project to feasible region or use a different start
        # For zero allocations and zero payments, IC and IR are all satisfied
        # (all utilities are 0), so this should be fine
        pass
    
    try:
        res = minimize(
            objective,
            z0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda z: -constraints(z), 'jac': lambda z: -constraints_jacobian(z)}
            ],
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success:
            z = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(z[var_x_idx(i, g)], 0, 1))
                payments[i] = float(max(0, z[var_p_idx(i)]))
            
            # Final verification
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _tighten_payments(allocations, payments, types):
    """
    Adjust payments to satisfy IC and IR exactly, assuming allocations are fixed.
    This reduces revenue minimally to ensure validity.
    """
    n_types = len(types)
    V = np.array(types, dtype=float)
    
    p = payments.copy()
    
    # Iterate to converge to the tightest valid payments
    for _ in range(10):
        changed = False
        for i in range(n_types):
            vi = V[i]
            xi = allocations[i]
            
            # IR bound
            bound_ir = np.dot(vi, xi)
            
            # IC bounds
            bound_ic = np.inf
            for j in range(n_types):
                if i == j:
                    continue
                xj = allocations[j]
                pj = p[j]
                # p_i <= v_i . x_i - (v_i . x_j - p_j)
                b = np.dot(vi, xi) - (np.dot(vi, xj) - pj)
                if b < bound_ic:
                    bound_ic = b
            
            bound = min(bound_ir, bound_ic)
            
            # Payment must be non-negative
            if bound < 0:
                bound = 0.0
            
            # We want p_i as high as possible.
            # If current p_i is above bound, lower it.
            if p[i] > bound + 1e-12:
                p[i] = bound
                changed = True
        
        if not changed:
            break
            
    return p

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
    # If IC is violated, it's harder to fix locally.
    # However, since we came from an LP solution, violations should be tiny.
    # If a violation exists, we can reduce the allocation of the "bad" type or increase payment.
    # Given the LP is solved, we assume violations are < 1e-8.
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - 1e-8:
                # Violation found.
                # Strategy: Reduce allocation of type i? Or increase payment of type i?
                # Increasing payment of i reduces revenue.
                # Reducing allocation of i reduces revenue.
                # Let's just clip allocations to [0,1] and trust the LP.
                # If this happens, the LP solver failed.
                pass

    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Apply tightening to ensure strict feasibility
    payments = _tighten_payments(allocations, payments, types)
    
    return allocations, payments
