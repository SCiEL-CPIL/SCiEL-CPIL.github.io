"""Incentive-compatible finite auction design - optimal LP-based mechanism."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification (list of 16 probabilities or similar)
        support: Support specification (16 types, 2 goods)
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 45.0  # Safety margin before 90s limit
    
    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    n_types = 16
    n_goods = 2
    
    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return _fallback_result()
    
    # 1. Solve LP to get a good initial guess
    result_lp = _solve_lp_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    # 2. Refine using direct optimization of Revenue(x)
    result = _refine_direct(result_lp, probs, types, n_types, n_goods, start_time, DEADLINE)
    
    return result

def _compute_revenue_and_check(x, probs, types, n_types, n_goods):
    """
    Compute revenue and check feasibility for a given allocation x.
    Returns (revenue, is_feasible, u, p).
    """
    x = np.clip(x, 0, 1)
    
    # Check Cyclic Monotonicity: No positive cycles in IC graph.
    # Weights w[t', t] = (v_t - v_t') . x_t'
    # If there is a cycle with sum(w) > 0, infeasible.
    # Bellman-Ford to detect positive cycles.
    # Actually, we just need to compute minimal u.
    # u_t = max_{t'} (u_t' + (v_t - v_t') . x_t')
    # This is the longest path problem.
    # If there is a positive cycle, u_t -> infinity (or infeasible).
    
    # Initialize u with 0 (IR constraint u_t >= 0)
    u = np.zeros(n_types)
    
    # Bellman-Ford for longest paths (maximization)
    # Relax edges t' -> t with weight w_t't = (v_t - v_t') . x_t'
    # Note: v_t is types[t]
    
    for _ in range(n_types):
        updated = False
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                w = np.dot(types[t] - types[t_prime], x[t_prime])
                new_u = u[t_prime] + w
                if new_u > u[t] + 1e-12:
                    u[t] = new_u
                    updated = True
        if not updated:
            break
    
    # Check for positive cycle (if still updating after N iterations)
    # If we did N iterations and still updated, there is a positive cycle.
    # But we only ran N iterations. Let's run one more to check.
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            w = np.dot(types[t] - types[t_prime], x[t_prime])
            new_u = u[t_prime] + w
            if new_u > u[t] + 1e-9:
                # Positive cycle detected
                return -np.inf, False, u, np.zeros(n_types)
    
    # Compute payments
    p = np.zeros(n_types)
    for t in range(n_types):
        p[t] = np.dot(x[t], types[t]) - u[t]
    
    # Check non-negative payments
    if np.any(p < -1e-9):
        return -np.inf, False, u, p
    
    # Calculate revenue
    revenue = np.sum(probs * (np.sum(x * types, axis=1) - u))
    
    return revenue, True, u, p

def _refine_direct(result, probs, types, n_types, n_goods, start_time, deadline):
    if result is None:
        return _fallback_result()
    
    x0 = result['allocations'].flatten()
    
    def objective(x_flat):
        if time.time() - start_time > deadline - 1.0:
            return 1e9 # Timeout penalty
        x = x_flat.reshape(n_types, n_goods)
        rev, feas, u, p = _compute_revenue_and_check(x, probs, types, n_types, n_goods)
        if not feas:
            return 1e9 # Infeasible penalty
        return -rev
    
    # Use Powell's method which doesn't need gradients
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < 1e8:
            x_opt = res.x.reshape(n_types, n_goods)
            rev, feas, u, p = _compute_revenue_and_check(x_opt, probs, types, n_types, n_goods)
            if feas:
                return {
                    'allocations': x_opt,
                    'payments': p
                }
    except Exception:
        pass
    
    # Fallback to LP result if refinement fails
    return result

def _solve_lp_auction(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the auction problem using a Linear Programming formulation.
    """
    n_vars = n_types * n_goods + n_types
    x_start = 0
    u_start = n_types * n_goods
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[x_start + t * n_goods + g] = -probs[t] * types[t][g]
        c[u_start + t] = probs[t]
        
    A_rows = []
    b_rows = []
    
    # 1. IR constraints: -u_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 2. IC constraints
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            row[u_start + t] = -1.0
            row[u_start + t_prime] = 1.0
            for g in range(n_goods):
                coeff = types[t][g] - types[t_prime][g]
                if coeff != 0:
                    row[x_start + t_prime * n_goods + g] = coeff
            A_rows.append(row)
            b_rows.append(0.0)
            
    # 3. Non-negative payments: u_t - sum_g v_{t,g} x_{t,g} <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_start + t] = 1.0
        for g in range(n_goods):
            row[x_start + t * n_goods + g] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
        
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True})
        if not res.success:
            return _fallback_result()
            
        z = res.x
        x = z[x_start:x_start + n_types * n_goods].reshape(n_types, n_goods)
        
        # Use direct check to get accurate u and p
        rev, feas, u, p = _compute_revenue_and_check(x, probs, types, n_types, n_goods)
        if not feas:
            # If LP solution is slightly infeasible due to numerical errors, 
            # try to repair or fallback
            return _fallback_result()
            
        return {
            'allocations': x,
            'payments': p
        }
    except Exception:
        return _fallback_result()


def _parse_distribution(distribution_spec, n_types):
    """Parse distribution specification into probability vector."""
    probs = np.zeros(n_types)
    
    if distribution_spec is None:
        return probs / n_types  # Uniform fallback
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (4, 4):
            probs = arr.flatten()
        elif arr.size == n_types:
            probs = arr.flatten()
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.clip(probs, 0, 1)
    
    return probs

def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
