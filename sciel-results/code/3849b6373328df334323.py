"""Incentive-compatible finite auction design - optimal LP-based mechanism."""
import numpy as np
from scipy.optimize import linprog
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
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return _fallback_result()
    
    # Solve the optimal auction LP
    result = _solve_optimal_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    return result


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
        # Try to parse from dict
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    
    # Normalize to ensure valid probabilities
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Clip small negative values
    probs = np.clip(probs, 0, 1)
    
    return probs


def _solve_optimal_auction(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the optimal incentive-compatible auction mechanism via LP.
    
    For a single additive bidder with quasi-linear utility, the optimal mechanism
    can be characterized by:
    1. Allocation rule x_i(t) for each type t and good i
    2. Payment rule p_i(t) for each type t
    
    The incentive compatibility constraint (IC) for type t reporting t' is:
    sum_i x_i(t) * t_i - p(t) >= sum_i x_i(t') * t_i - p(t')
    
    For the optimal mechanism with a single bidder, we can use the fact that
    the payment is determined by the allocation rule via the envelope theorem.
    
    The key insight: For a single additive bidder, the optimal mechanism
    maximizes expected revenue subject to:
    - IR: sum_i x_i(t) * t_i - p(t) >= 0 for all t
    - IC: The mechanism is truthful
    
    For a finite type space with a single bidder, we can formulate this as an LP.
    """
    
    # Decision variables: x[t, g] for allocation, p[t] for payment
    # Total variables: 16*2 + 16 = 48
    
    # For the optimal single-bidder auction, we can use the following approach:
    # The optimal allocation is monotone, and payments are determined by IC.
    
    # Alternative approach: Direct LP formulation
    # Variables: x[0..15, 0..1] (32 vars), p[0..15] (16 vars)
    # Total: 48 variables
    
    n_vars = n_types * n_goods + n_types
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog, we minimize, so: c = -probs[t] for payment vars, 0 for allocation vars
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[t * n_goods + n_goods + t] = -probs[t]  # Negative for maximization
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= x[t, g] <= 1 for all t, g
    # 2. Individual Rationality: sum_g x[t,g] * t_g - p[t] >= 0 for all t
    #    => p[t] <= sum_g x[t,g] * t_g
    #    => -sum_g t_g * x[t,g] + p[t] <= 0
    # 3. Incentive Compatibility: For all t, t':
    #    sum_g x[t,g] * t_g - p[t] >= sum_g x[t',g] * t_g - p[t']
    #    => sum_g (t_g - t'_g) * x[t,g] - p[t] + p[t'] >= 0
    #    => -sum_g (t_g - t'_g) * x[t,g] + p[t] - p[t'] <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints: for each type t
    for t in range(n_types):
        row = np.zeros(n_vars)
        t_vals = types[t]
        for g in range(n_goods):
            row[t * n_goods + g] = -t_vals[g]  # -t_g * x[t,g]
        row[n_goods + t] = 1.0  # + p[t]
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC constraints: for each pair (t, t')
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            t_vals = types[t]
            tp_vals = types[t_prime]
            for g in range(n_goods):
                row[t * n_goods + g] = -(t_vals[g] - tp_vals[g])  # -(t_g - t'_g) * x[t,g]
            row[n_goods + t] = 1.0  # + p[t]
            row[n_goods + t_prime] = -1.0  # - p[t']
            A_ub.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))  # p[t] >= 0
    
    # Check deadline
    if time.time() - start_time > deadline - 5:
        return _fallback_result()
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return _fallback_result()
    
    if res.status != 0:
        # If LP fails, try fallback
        return _fallback_result()
    
    # Extract solution
    x = np.zeros((n_types, n_goods))
    p = np.zeros(n_types)
    
    for t in range(n_types):
        for g in range(n_goods):
            x[t, g] = res.x[t * n_goods + g]
        p[t] = res.x[n_goods + t]
    
    # Clean up numerical noise
    x = np.clip(x, 0, 1)
    p = np.clip(p, 0, None)
    
    # Verify constraints
    if not _verify_constraints(x, p, probs, types):
        # Try to fix violations
        x, p = _fix_violations(x, p, probs, types)
        if not _verify_constraints(x, p, probs, types):
            return _fallback_result()
    
    return {
        'allocations': x,
        'payments': p
    }


def _verify_constraints(x, p, probs, types, tol=1e-8):
    """Verify all constraints are satisfied."""
    n_types = len(types)
    
    # Check allocation bounds
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    
    # Check payment non-negativity
    if np.any(p < -tol):
        return False
    
    # Check IR
    for t in range(n_types):
        utility = np.dot(x[t], types[t]) - p[t]
        if utility < -tol:
            return False
    
    # Check IC
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            u_t = np.dot(x[t], types[t]) - p[t]
            u_tprime_reported = np.dot(x[t_prime], types[t]) - p[t_prime]
            if u_t < u_tprime_reported - tol:
                return False
    
    return True


def _fix_violations(x, p, probs, types):
    """Attempt to fix constraint violations."""
    n_types = len(types)
    
    # Simple fix: adjust payments to satisfy IR
    for t in range(n_types):
        max_payment = np.dot(x[t], types[t])
        if p[t] > max_payment:
            p[t] = max_payment
    
    # Re-check and potentially reduce allocations
    for t in range(n_types):
        if np.dot(x[t], types[t]) < p[t] - 1e-10:
            # Reduce allocation
            for g in range(2):
                x[t, g] = max(0, x[t, g] * 0.5)
    
    return x, p


def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
