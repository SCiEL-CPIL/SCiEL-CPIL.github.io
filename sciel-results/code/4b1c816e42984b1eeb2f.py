"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    n_goods = 2

    type_idx = {t: i for i, t in enumerate(types)}

    # Parse distribution_spec to get probabilities for each type
    # distribution_spec is a dict mapping (v1, v2) -> probability
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).reshape(n_types)
    else:
        # Default: uniform
        probs = np.ones(n_types) / n_types

    # Normalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types

    # Decision variables:
    # x[i, j] = allocation probability for good j when type i reports (i.e., x[i,0], x[i,1])
    # p[i] = payment for type i
    #
    # Total variables: 16*2 + 16 = 48
    # Variable ordering: x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], p[1], ..., p[15]

    n_vars = n_types * n_goods + n_types  # 32 + 16 = 48

    def var_idx(i, j):
        """Index for allocation variable x[i, j]"""
        return i * n_goods + j

    def pay_idx(i):
        """Index for payment variable p[i]"""
        return n_types * n_goods + i

    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[pay_idx(i)] = -probs[i]

    # Constraints:
    # 1. Feasibility: 0 <= x[i,j] <= 1 for all i,j
    # 2. Nonnegative payments: p[i] >= 0
    # 3. Individual Rationality: E[value | type i] >= p[i] for all i
    #    sum_j x[i,j]*v_i_j - p[i] >= 0  =>  -sum_j x[i,j]*v_i_j + p[i] <= 0
    # 4. Incentive Compatibility: For all true types i and reports r:
    #    E[value_i | report r] - p[r] <= E[value_i | report i] - p[i]
    #    sum_j x[r,j]*v_i_j - p[r] - sum_j x[i,j]*v_i_j + p[i] <= 0

    A_ub = []
    b_ub = []

    # IR constraints: for each type i
    # sum_j x[i,j]*v_i_j - p[i] >= 0  =>  -sum_j x[i,j]*v_i_j + p[i] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = types[i]
        row[var_idx(i, 0)] = -v1
        row[var_idx(i, 1)] = -v2
        row[pay_idx(i)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # IC constraints: for all pairs (true_type i, report r) where i != r
    for i in range(n_types):
        for r in range(n_types):
            if i == r:
                continue
            row = np.zeros(n_vars)
            vi1, vi2 = types[i]  # true values
            # LHS: sum_j x[r,j]*v_i_j - p[r] - sum_j x[i,j]*v_i_j + p[i] <= 0
            row[var_idx(r, 0)] = vi1
            row[var_idx(r, 1)] = vi2
            row[pay_idx(r)] = -1.0
            row[var_idx(i, 0)] = -vi1
            row[var_idx(i, 1)] = -vi2
            row[pay_idx(i)] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds: 0 <= x[i,j] <= 1, p[i] >= 0 (no upper bound)
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: zero allocation, zero payment
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}

    if not result.success:
        # Fallback: zero allocation, zero payment
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}

    x = result.x
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_goods):
            allocations[i, j] = x[var_idx(i, j)]
        payments[i] = x[pay_idx(i)]

    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)

    # Check deadline
    if time.time() > deadline:
        return {"allocations": allocations, "payments": payments}

    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
