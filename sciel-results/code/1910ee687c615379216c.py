"""Incentive-compatible finite auction design with LP-based optimal mechanism."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin

    # Parse the distribution
    # distribution_spec should give us the probability distribution over the 16 reports
    # support: list of (v1, v2) pairs, or we assume standard {0,1,2,3} x {0,1,2,3}
    # constraints: dict with various constraints

    # Define the report space
    values_list = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_reports = 16
    n_goods = 2

    # Map report index to value vector
    def idx(v1, v2):
        return v1 * 4 + v2

    # Parse distribution_spec to get probabilities
    probs = np.ones(n_reports) / n_reports  # default uniform
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            p = distribution_spec['probs']
            if len(p) == n_reports:
                probs = np.array(p, dtype=float)
            elif isinstance(p, dict):
                for k, v in p.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        probs[idx(k[0], k[1])] = v
                    elif isinstance(k, int) and 0 <= k < n_reports:
                        probs[k] = v
        elif 'type' in distribution_spec:
            dist_type = distribution_spec['type']
            if dist_type == 'uniform':
                probs = np.ones(n_reports) / n_reports
            elif dist_type == 'independent':
                # Each good independent uniform on {0,1,2,3}
                probs = np.ones(n_reports) / n_reports
            elif dist_type == 'correlated':
                if 'matrix' in distribution_spec:
                    mat = np.array(distribution_spec['matrix'], dtype=float)
                    probs = mat.flatten()
            elif dist_type == 'custom':
                if 'probs' in distribution_spec:
                    probs = np.array(distribution_spec['probs'], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_reports:
            probs = np.array(distribution_spec, dtype=float)

    # Normalize probabilities
    probs = probs / probs.sum()

    # Value vectors for each report
    V = np.array([[v1, v2] for v1 in range(4) for v2 in range(4)], dtype=float)

    # Variables: x[0..31] for allocations (16 reports x 2 goods), p[32..47] for payments
    # x[i*2 + j] = allocation of good j at report i
    # p[i] = payment at report i
    n_vars = n_reports * n_goods + n_reports  # 32 + 16 = 48

    def x_var(i, j):
        return i * n_goods + j

    def p_var(i):
        return n_reports * n_goods + i

    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # In linprog, we minimize, so c = -probs for payment vars, 0 for allocation vars
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[p_var(i)] = -probs[i]

    # Constraints:
    # 1. Allocation feasibility: 0 <= x[i,j] <= 1 for all i,j
    # 2. IC constraints: For each pair (i,j), u_i >= u_j where u_k = V[k]·x_k - p_k
    #    V[i]·x_i - p_i >= V[i]·x_j - p_j
    #    => sum_j V[i,j]*x[i,j] - p[i] - sum_j V[i,j]*x[j,j] + p[j] >= 0
    #    => -sum_j V[i,j]*x[i,j] + p[i] + sum_j V[i,j]*x[j,j] - p[j] <= 0
    # 3. IR constraints: V[i]·x_i - p[i] >= 0
    #    => -sum_j V[i,j]*x[i,j] + p[i] <= 0

    A_ub = []
    b_ub = []

    # IC constraints: for each ordered pair (i, j), i != j
    # V[i]·x_i - p_i >= V[i]·x_j - p_j
    # => sum_j V[i,j]*x[i,j] - sum_j V[i,j]*x[j,j] - p[i] + p[j] >= 0
    # Multiply by -1: -sum_j V[i,j]*x[i,j] + sum_j V[i,j]*x[j,j] + p[i] - p[j] <= 0
    for i in range(n_reports):
        for j in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -V[i,j] * x[i,j] for each good j
            for g in range(n_goods):
                row[x_var(i, g)] = -V[i, g]
                row[x_var(j, g)] = V[i, g]
            row[p_var(i)] = 1.0
            row[p_var(j)] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: V[i]·x_i - p[i] >= 0
    # => -sum_g V[i,g]*x[i,g] + p[i] <= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[x_var(i, g)] = -V[i, g]
        row[p_var(i)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: allocations in [0,1], payments in [0, inf)
    bounds = []
    for i in range(n_reports):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_reports):
        bounds.append((0, None))

    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
    except Exception:
        # Fallback: simple mechanism
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)
        return {'allocations': allocations, 'payments': payments}

    x = result.x

    # Extract allocations and payments
    allocations = np.zeros((n_reports, n_goods))
    payments = np.zeros(n_reports)
    for i in range(n_reports):
        for g in range(n_goods):
            allocations[i, g] = x[x_var(i, g)]
        payments[i] = x[p_var(i)]

    # Clip small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)

    # Verify constraints with tolerance
    def verify():
        if time.time() > deadline:
            return False
        # Check IC
        for i in range(n_reports):
            for j in range(n_reports):
                u_i = V[i] @ allocations[i] - payments[i]
                u_j = V[i] @ allocations[j] - payments[j]
                if u_i < u_j - 1e-8:
                    return False
        # Check IR
        for i in range(n_reports):
            u_i = V[i] @ allocations[i] - payments[i]
            if u_i < -1e-8:
                return False
        # Check feasibility
        if np.any(allocations < -1e-8) or np.any(allocations > 1 + 1e-8):
            return False
        if np.any(payments < -1e-8):
            return False
        return True

    if not verify():
        # Try to fix small violations
        # Re-solve with tighter constraints or use a simpler approach
        # For now, return zero mechanism as safe fallback
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
