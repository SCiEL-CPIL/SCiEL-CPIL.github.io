"""Incentive-compatible finite auction design via LP over allocation probabilities and utilities."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict or object with distribution information.
        support: List of (v1, v2) tuples representing the value support.
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = 90.0  # wall-clock limit
    
    # Parse the support: expect 16 type profiles (v1, v2) with v1, v2 in {0,1,2,3}
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Ensure support has exactly 16 entries
    if len(support) != 16:
        # Try to construct standard support
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Parse distribution: expect probabilities over the 16 type profiles
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        else:
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    if len(probs) != 16:
        probs = np.ones(16) / 16.0
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(16) / 16.0
    
    n_types = 16
    n_goods = 2
    
    # Values for each type profile
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # The mechanism design problem:
    # Variables:
    #   x[i, j] = probability of allocating good j to type i (i in 0..15, j in 0..1)
    #   u[i] = utility of type i
    #   p[i] = payment of type i
    #
    # For a single additive bidder with two goods, the utility of type i with value vector v_i
    # is: u_i = sum_j x[i,j] * v_i[j] - p[i]
    #
    # Incentive compatibility (IC) for single bidder:
    # For all i, i': u_i >= sum_j x[i',j] * v_i[j] - p[i']
    # This is equivalent to: u_i - sum_j x[i,j]*v_i[j] <= u_i' - sum_j x[i',j]*v_i[j]
    # i.e., p_i - sum_j x[i,j]*v_i[j] <= p_i' - sum_j x[i',j]*v_i[j]
    #
    # Revenue = E[p_i] = sum_i probs[i] * p[i]
    #
    # We can reformulate using the envelope theorem approach:
    # Let w[i] = p[i] - sum_j x[i,j]*v_i[j] (the "payment minus value-weighted allocation")
    # Then IC requires: w[i] <= w[i'] for all i, i' where type i can mimic type i'
    # Actually, IC is: u_i >= u_i' + (v_i - v_i') dot x[i']
    # => sum_j x[i,j]*v_i[j] - p[i] >= sum_j x[i',j]*v_i[j] - p[i']
    # => p[i] - sum_j x[i,j]*v_i[j] <= p[i'] - sum_j x[i',j]*v_i[j]
    # => w[i] <= w[i']
    #
    # So w must be non-decreasing in the sense that w[i] <= w[i'] whenever... 
    # Actually for all pairs, w[i] <= w[i'] must hold for the IC constraint to be satisfied
    # when type i reports i'. Wait, let me re-derive.
    #
    # IC: For all true type i and reported type i':
    #   u_i >= v_i · x[i'] - p[i']
    #   sum_j x[i,j]*v_i[j] - p[i] >= sum_j x[i',j]*v_i[j] - p[i']
    #   (p[i'] - sum_j x[i',j]*v_i[j]) - (p[i] - sum_j x[i,j]*v_i[j]) <= 0
    #   w[i'] evaluated at v_i minus w[i] evaluated at v_i... this isn't quite right.
    #
    # Let me use the standard formulation. Define:
    #   U(i) = expected utility of type i when telling the truth = v_i · x[i] - p[i]
    # IC: U(i) >= v_i · x[i'] - p[i'] for all i, i'
    #
    # Revenue = E[p(i)] = E[v_i · x[i]] - E[U(i)]
    #
    # We want to maximize E[p(i)] = sum_i pi_i * p(i)
    # subject to:
    #   0 <= x[i,j] <= 1 for all i, j
    #   p(i) >= 0 for all i (nonnegative payments)
    #   U(i) >= v_i · x[i'] - p(i') for all i, i' (IC)
    #   U(i) >= 0 for all i (individual rationality)
    #
    # Variables: x[i,j] (32 vars), U(i) (16 vars), p(i) (16 vars) = 64 vars
    #
    # Let's set up the LP. Maximize sum_i pi_i * p(i)
    #
    # Constraints:
    # 1. x[i,j] >= 0: -x[i,j] <= 0
    # 2. x[i,j] <= 1: x[i,j] <= 1
    # 3. p(i) >= 0: -p(i) <= 0
    # 4. U(i) >= 0: -U(i) <= 0
    # 5. IC: U(i) - v_i · x[i'] + p(i') >= 0 for all i, i'
    #    => -U(i) + v_i · x[i'] - p(i') <= 0
    
    # Variable ordering: x[0,0], x[0,1], ..., x[15,0], x[15,1], U[0], ..., U[15], p[0], ..., p[15]
    # Total: 32 + 16 + 16 = 64 variables
    
    n_x = n_types * n_goods  # 32
    n_u = n_types  # 16
    n_p = n_types  # 16
    n_vars = n_x + n_u + n_p  # 64
    
    def x_idx(i, j):
        return i * n_goods + j
    
    def u_idx(i):
        return n_x + i
    
    def p_idx(i):
        return n_x + n_u + i
    
    # Objective: maximize sum_i pi_i * p(i)
    # linprog minimizes, so we minimize -sum_i pi_i * p(i)
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_vals = []
    
    # 1. x[i,j] <= 1
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[x_idx(i, j)] = 1.0
            A_rows.append(row)
            b_vals.append(1.0)
    
    # 2. -x[i,j] <= 0 (x[i,j] >= 0)
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[x_idx(i, j)] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # 3. -p(i) <= 0 (p(i) >= 0)
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(i)] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # 4. -U(i) <= 0 (U(i) >= 0)
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(i)] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # 5. IC: -U(i) + v_i · x[i'] - p(i') <= 0 for all i, i'
    for i in range(n_types):
        for i_prime in range(n_types):
            row = np.zeros(n_vars)
            row[u_idx(i)] = -1.0
            # v_i · x[i'] = sum_j v_i[j] * x[i', j]
            for j in range(n_goods):
                row[x_idx(i_prime, j)] = values[i][j]
            row[p_idx(i_prime)] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: all variables >= 0, x[i,j] <= 1, U and p unbounded above (but constrained)
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))  # U(i) >= 0
    for i in range(n_types):
        bounds.append((0.0, None))  # p(i) >= 0
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception as e:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if not result.success:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x[:n_x].reshape(n_types, n_goods)
    u_opt = result.x[n_x:n_x + n_u]
    p_opt = result.x[n_x + n_u:n_x + n_u + n_p]
    
    # Check time
    if time.time() > deadline - 5.0:
        return {'allocations': x_opt, 'payments': p_opt}
    
    # Verify IC and IR constraints, fix small violations
    eps = 1e-8
    for i in range(n_types):
        # IR: U(i) >= 0
        if u_opt[i] < -eps:
            u_opt[i] = 0.0
            # Recompute p[i] from U(i) = v_i · x[i] - p[i]
            p_opt[i] = values[i] @ x_opt[i] - u_opt[i]
    
    # Recompute payments from utilities to ensure consistency
    # U(i) = v_i · x[i] - p[i] => p[i] = v_i · x[i] - U(i)
    for i in range(n_types):
        p_opt[i] = values[i] @ x_opt[i] - u_opt[i]
        if p_opt[i] < 0:
            p_opt[i] = 0.0
            u_opt[i] = values[i] @ x_opt[i] - p_opt[i]
    
    # Verify IC constraints and fix if needed
    # U(i) >= v_i · x[i'] - p(i') for all i, i'
    for i in range(n_types):
        for i_prime in range(n_types):
            lhs = u_opt[i]
            rhs = values[i] @ x_opt[i_prime] - p_opt[i_prime]
            if lhs < rhs - eps:
                # Violation; adjust U(i) and p(i)
                u_opt[i] = rhs
                p_opt[i] = values[i] @ x_opt[i] - u_opt[i]
                if p_opt[i] < 0:
                    p_opt[i] = 0.0
                    u_opt[i] = values[i] @ x_opt[i]
    
    # Final check: ensure nonnegative payments
    p_opt = np.maximum(p_opt, 0.0)
    
    # Ensure allocations are in [0, 1]
    x_opt = np.clip(x_opt, 0.0, 1.0)
    
    return {'allocations': x_opt, 'payments': p_opt}
# EVOLVE-BLOCK-END
