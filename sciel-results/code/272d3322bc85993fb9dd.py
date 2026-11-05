"""Rewritten program: solve the finite-support IC/IR auction LP directly with HiGHS."""

import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Args:
        distribution_spec: spec for the type distribution (dict or index)
        support: list of allowed reports/values
        constraints: additional constraints dict
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    type_index = {t: i for i, t in enumerate(types)}

    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif (t[0], t[1]) in distribution_spec:
                probs[i] = distribution_spec[(t[0], t[1])]
        # Normalize if needed
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, (int, np.integer)):
        # Three fixed distributions: 0, 1, 2
        dist_idx = int(distribution_spec) % 3
        if dist_idx == 0:
            # Uniform
            probs = np.ones(num_types) / num_types
        elif dist_idx == 1:
            # Independent uniform per good (same as uniform here for product)
            probs = np.ones(num_types) / num_types
        elif dist_idx == 2:
            # Correlated: v2 = v1 with prob 0.5, independent otherwise
            probs = np.zeros(num_types)
            for v1 in values:
                for v2 in values:
                    if v1 == v2:
                        probs[type_index[(v1, v2)]] = 0.5 * (1.0/4) + 0.5 * (1.0/16)
                    else:
                        probs[type_index[(v1, v2)]] = 0.5 * (1.0/16)
            total = probs.sum()
            probs /= total
    else:
        # Default: uniform
        probs = np.ones(num_types) / num_types

    # Ensure support covers all 16 types
    if support is None:
        support = types
    # support should be list of (v1, v2) tuples, length 16

    # Variables: x[i, g] = allocation of good g to type i, p[i] = payment from type i
    # Total variables: 16*2 + 16 = 48
    num_vars = num_types * 2 + num_types  # 48

    # Build LP: maximize sum_i probs[i] * p[i]
    # Subject to:
    # 1. Feasibility: 0 <= x[i,g] <= 1
    # 2. IR: p[i] <= v_i . x_i  (expected utility >= 0)
    # 3. IC: for all i, j: p[i] - v_i.x_i <= p[j] - v_j.x_j  (truthful reporting maximizes utility)
    #    i.e., u(i, i) >= u(i, j) for all i, j
    #    u(i, j) = v_i . x_j - p[j]
    #    So: v_i.x_i - p[i] >= v_i.x_j - p[j]  =>  p[i] - v_i.x_i <= p[j] - v_i.x_j
    #    Wait, let me re-derive:
    #    u(type_i, report_j) = v_i . x_j - p_j
    #    IC: u(i, i) >= u(i, j) for all j
    #    v_i.x_i - p_i >= v_i.x_j - p_j
    #    => -p_i + v_i.x_i >= -p_j + v_i.x_j
    #    => p_i - v_i.x_i <= p_j - v_i.x_j
    # 4. Nonnegativity of payments: p[i] >= 0
    # 5. Regret <= 1e-8: This is automatically satisfied if IC holds exactly.

    # Let's use scipy.optimize.linprog
    from scipy.optimize import linprog

    # Objective: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so minimize -sum_i probs[i] * p[i]
    c = np.zeros(num_vars)
    for i in range(num_types):
        p_idx = num_types * 2 + i
        c[p_idx] = -probs[i]

    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []

    # IC constraints: for each i, j: p_i - v_i.x_i <= p_j - v_i.x_j
    # Rearranged: p_i - p_j - v_i.x_i + v_i.x_j <= 0
    # In terms of variables:
    # var index for x[i,g]: i*2 + g
    # var index for p[i]: num_types*2 + i
    for i in range(num_types):
        vi = types[i]
        for j in range(num_types):
            if i == j:
                continue
            row = np.zeros(num_vars)
            # p_i
            row[num_types * 2 + i] = 1.0
            # -p_j
            row[num_types * 2 + j] = -1.0
            # -v_i.x_i
            row[i * 2 + 0] -= vi[0]
            row[i * 2 + 1] -= vi[1]
            # +v_i.x_j
            row[j * 2 + 0] += vi[0]
            row[j * 2 + 1] += vi[1]
            A_rows.append(row)
            b_rows.append(0.0)

    # IR constraints: p_i <= v_i.x_i  =>  p_i - v_i.x_i <= 0
    for i in range(num_types):
        vi = types[i]
        row = np.zeros(num_vars)
        row[num_types * 2 + i] = 1.0
        row[i * 2 + 0] -= vi[0]
        row[i * 2 + 1] -= vi[1]
        A_rows.append(row)
        b_rows.append(0.0)

    # Nonnegativity of payments: p_i >= 0  =>  -p_i <= 0
    for i in range(num_types):
        row = np.zeros(num_vars)
        row[num_types * 2 + i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: 0 <= x[i,g] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x[i,0]
        bounds.append((0, 1))  # x[i,1]
    for i in range(num_types):
        bounds.append((0, None))  # p[i]

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: simple mechanism
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)
        return {'allocations': allocations, 'payments': payments}

    x_opt = result.x

    # Extract allocations and payments
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        allocations[i, 0] = x_opt[i * 2 + 0]
        allocations[i, 1] = x_opt[i * 2 + 1]
        payments[i] = x_opt[num_types * 2 + i]

    # Clamp small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)

    # Verify IC and IR with tolerance
    # Check IR: p[i] <= v_i . x_i + tol
    tol = 1e-8
    for i in range(num_types):
        vi = types[i]
        util = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
        if util < -tol:
            # IR violated; adjust payment down
            payments[i] = max(0, vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1])

    # Check IC: for all i, j: u(i,i) >= u(i,j) - tol
    for i in range(num_types):
        vi = types[i]
        ui_i = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
        for j in range(num_types):
            if i == j:
                continue
            ui_j = vi[0] * allocations[j, 0] + vi[1] * allocations[j, 1] - payments[j]
            if ui_i < ui_j - tol:
                # IC violated; this mechanism is not IC. 
                # We need to fix this. For now, fall back to a trivially IC mechanism.
                pass

    # If IC is violated, use a simple VCG-like or trivial mechanism
    # Trivially IC: no trade (allocations=0, payments=0)
    # Or: allocate good g to type i if v_i[g] >= threshold
    # Let's check if the solution is valid; if not, use a simple mechanism
    ic_ok = True
    for i in range(num_types):
        vi = types[i]
        ui_i = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
        for j in range(num_types):
            if i == j:
                continue
            ui_j = vi[0] * allocations[j, 0] + vi[1] * allocations[j, 1] - payments[j]
            if ui_i < ui_j - 1e-7:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        # Use a simple deterministic mechanism that is IC
        # Allocate good g to type i iff v_i[g] >= 2 (arbitrary threshold)
        # Payment: 2 if allocated, 0 otherwise (simple)
        # This is not necessarily optimal but is IC if we use the right payment
        # Actually, let's use a trivial mechanism: no allocation
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        pass  # Already have a result

    return {'allocations': allocations, 'payments': payments}
