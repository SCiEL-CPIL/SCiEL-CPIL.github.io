"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.array(distribution_spec.get('values', [1/16]*16), dtype=float)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Parse support
    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    n_reports = len(support_list)
    if n_reports != 16:
        # Pad or truncate to 16
        if n_reports < 16:
            support_list = support_list + [(3, 3)] * (16 - n_reports)
            probs = np.concatenate([probs, np.zeros(16 - len(probs))])
        else:
            support_list = support_list[:16]
            probs = probs[:16]
    
    # Build index mapping
    value_to_idx = {}
    for idx, (v1, v2) in enumerate(support_list):
        value_to_idx[(v1, v2)] = idx
    
    # All possible values in {0,1,2,3}^2
    all_values = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Ensure we have all 16 values
    for val in all_values:
        if val not in value_to_idx:
            # This shouldn't happen if support is correct
            pass
    
    # Variables: x[i][j] for i in 0..15 (report), j in 0..1 (good)
    # Total variables: 16 * 2 = 32
    # Plus payment variables: p[i] for i in 0..15
    # Total: 32 + 16 = 48
    
    # For incentive compatibility (truthfulness), we use the payment identity:
    # p(i) = v_i · x(i) - ∫_0^{v_i} x'(t) dt for continuous types
    # For finite types, we use the discrete version:
    # p(i) = v_i · x(i) - sum over lower values of (v_i - v_j) · x(j)
    # 
    # Actually, for finite types with values in {0,1,2,3}, the IC constraint is:
    # For each report i and each alternative report j:
    # v_i · x(i) - p(i) >= v_i · x(j) - p(j)
    # 
    # This means: p(i) - p(j) <= v_i · (x(i) - x(j))
    # 
    # We can eliminate payments by expressing them in terms of allocations.
    # Set p(0) = 0 (without loss of generality, since adding a constant to all payments
    # doesn't change incentives but affects revenue; we'll optimize over the constant).
    # 
    # Actually, let's keep payments as variables and add IC constraints directly.
    # 
    # Variables:
    # x[0..15][0..1] : allocation probabilities (32 vars)
    # p[0..15] : payments (16 vars)
    # Total: 48 vars
    #
    # Objective: Maximize sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. 0 <= x[i][j] <= 1 for all i, j
    # 2. p[i] >= 0 for all i (nonnegative payments)
    # 3. IR: v_i · x(i) - p(i) >= 0 for all i
    # 4. IC: v_i · x(i) - p(i) >= v_i · x(j) - p(j) for all i, j
    #    i.e., p(i) - p(j) <= v_i · (x(i) - x(j))
    # 5. Regret <= 1e-8 (this is the IC constraint essentially)
    
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p
    
    # Objective: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for i in range(16):
        c[n_x + i] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Constraint 1: x[i][j] <= 1
    for i in range(16):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # Constraint 2: -x[i][j] <= 0 (i.e., x[i][j] >= 0)
    for i in range(16):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # Constraint 3: -p[i] <= 0 (i.e., p[i] >= 0)
    for i in range(16):
        row = np.zeros(n_vars)
        row[n_x + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Constraint 4: IR: v_i · x(i) - p(i) >= 0
    # i.e., -v_i · x(i) + p(i) <= 0
    for i in range(16):
        v1, v2 = support_list[i]
        row = np.zeros(n_vars)
        row[i * 2 + 0] = -v1
        row[i * 2 + 1] = -v2
        row[n_x + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Constraint 5: IC: p(i) - p(j) <= v_i · (x(i) - x(j))
    # i.e., p(i) - p(j) - v_i · x(i) + v_i · x(j) <= 0
    for i in range(16):
        v1_i, v2_i = support_list[i]
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = support_list[j]
            row = np.zeros(n_vars)
            # p(i) term
            row[n_x + i] = 1.0
            # -p(j) term
            row[n_x + j] = -1.0
            # -v_i · x(i) term
            row[i * 2 + 0] = -v1_i
            row[i * 2 + 1] = -v2_i
            # +v_i · x(j) term
            row[j * 2 + 0] = v1_i
            row[j * 2 + 1] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: all variables >= 0, x[i][j] <= 1, p[i] unbounded above (but constrained by IR)
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        result = None
    
    # Check time
    if time.time() > deadline:
        # Return a simple fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result is None or not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        # Give both goods with probability 0.5, charge average value
        for i in range(16):
            v1, v2 = support_list[i]
            allocations[i] = [0.5, 0.5]
            payments[i] = 0.5 * (v1 + v2) * 0.5  # Simple payment
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_vals = result.x[:n_x].reshape(16, 2)
    p_vals = result.x[n_x:]
    
    # Clip to valid range
    x_vals = np.clip(x_vals, 0.0, 1.0)
    p_vals = np.clip(p_vals, 0.0, None)
    
    # Verify constraints and fix any violations
    for i in range(16):
        v1, v2 = support_list[i]
        # IR check
        utility = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - p_vals[i]
        if utility < -1e-10:
            # Fix by reducing payment
            p_vals[i] = v1 * x_vals[i, 0] + v2 * x_vals[i, 1]
    
    # Check IC constraints
    for i in range(16):
        v1_i, v2_i = support_list[i]
        for j in range(16):
            if i == j:
                continue
            # Check: v_i · x(i) - p(i) >= v_i · x(j) - p(j)
            lhs = v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - p_vals[i]
            rhs = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - p_vals[j]
            if lhs < rhs - 1e-8:
                # IC violation; adjust payment
                # p(i) should be at most v_i · x(i) - v_i · x(j) + p(j)
                p_vals[i] = min(p_vals[i], v1_i * x_vals[i, 0] + v2_i * x_vals[i, 1] - 
                               v1_i * x_vals[j, 0] - v2_i * x_vals[j, 1] + p_vals[j])
                p_vals[i] = max(p_vals[i], 0.0)
    
    return {'allocations': x_vals, 'payments': p_vals}
# EVOLVE-BLOCK-END
