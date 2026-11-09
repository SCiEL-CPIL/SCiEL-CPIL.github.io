"""Rewritten program: direct LP over allocation probabilities with IC-embedded payments."""

def construct_auction(distribution_spec, support, constraints):
    import time
    import itertools
    import numpy as np
    from scipy.optimize import linprog

    t0 = time.time()
    DEADLINE = 85.0  # 90s wall-clock, 5s margin

    # Parse distribution_spec to get the probability distribution over types
    # distribution_spec is expected to describe the probability mass function
    # support: list of value pairs (v1, v2) for the two goods
    # We need to enumerate all allowed reports/types

    # Determine the type set from support
    # support could be a list of tuples or a specification
    # Let's handle common formats

    if isinstance(distribution_spec, dict):
        # Could be {'probs': [...], 'types': [...]} or similar
        if 'probs' in distribution_spec and 'types' in distribution_spec:
            types = list(distribution_spec['types'])
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'support' in distribution_spec:
            types = list(support)
            if 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float)
            else:
                probs = np.ones(len(types), dtype=float) / len(types)
        else:
            types = list(support)
            probs = np.ones(len(types), dtype=float) / len(types)
    elif isinstance(distribution_spec, (list, tuple)):
        # Maybe it's the probs directly
        if len(distribution_spec) == len(support):
            types = list(support)
            probs = np.array(distribution_spec, dtype=float)
        else:
            types = list(support)
            probs = np.ones(len(types), dtype=float) / len(types)
    else:
        types = list(support)
        probs = np.ones(len(types), dtype=float) / len(types)

    n = len(types)

    # Extract values: each type is (v1, v2)
    V = np.zeros((n, 2), dtype=float)
    for i, t in enumerate(types):
        V[i, 0] = float(t[0])
        V[i, 1] = float(t[1])

    # Normalize probs
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = np.ones(n, dtype=float) / n

    # For a single additive bidder with two goods:
    # Type i has value vector v_i = (v_i1, v_i2)
    # Allocation: q_i = (q_i1, q_i2) where q_ij in [0,1]
    # Payment: p_i
    #
    # IC condition: For all types i and reports j:
    #   v_i . q_i - p_i >= v_i . q_j - p_j
    #
    # IR condition: v_i . q_i - p_i >= 0 for all i
    #
    # Objective: Maximize sum_i probs[i] * p_i
    #
    # The IC constraints imply that the payment is determined up to a constant
    # by the allocation rule. Specifically, for any type i:
    #   p_i = v_i . q_i - sum_j probs[j] * (v_j . q_j) * indicator-related terms
    #
    # Actually, let's use the standard result: for a single bidder with discrete types,
    # IC is equivalent to:
    #   p_i = v_i . q_i - sum_{j: v_j <= v_i in some sense} ... 
    #
    # For multi-dimensional types, it's more complex. Let's just set up the full LP
    # with explicit IC constraints since n is small (16 types max based on task).
    #
    # Variables: q_11, q_12, ..., q_1n1, q_1n2, ..., p_1, p_2, ..., p_n
    # That's 2n + n = 3n variables.
    #
    # Actually, let's think about this more carefully. The task says 16 allocations 
    # and 16 payments, so n=16 types.
    #
    # Variables: for each type i, allocation (q_i1, q_i2) and payment p_i.
    # Total variables: 3n = 48 for n=16.
    #
    # Constraints:
    # 1. Feasibility: 0 <= q_ij <= 1 for all i, j
    # 2. Nonnegative payments: p_i >= 0
    # 3. IR: v_i . q_i - p_i >= 0  =>  p_i <= v_i . q_i
    # 4. IC: v_i . q_i - p_i >= v_i . q_j - p_j  for all i, j
    #    => v_i . (q_i - q_j) + (p_j - p_i) >= 0
    #
    # Objective: Maximize sum_i probs[i] * p_i
    #
    # This is a linear program! Let's solve it directly.

    # Build the LP
    # Variable ordering: [q_11, q_12, q_21, q_22, ..., q_n1, q_n2, p_1, p_2, ..., p_n]
    n_vars = 3 * n
    q_start = lambda i: 2 * i  # index of q_i1
    q1_idx = lambda i: 2 * i
    q2_idx = lambda i: 2 * i + 1
    p_idx = lambda i: 2 * n + i

    # Objective: maximize sum_i probs[i] * p_i
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(n):
        c[p_idx(i)] = -probs[i]

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_rows = []

    # Feasibility: q_ij <= 1  (lower bound 0 handled by bounds)
    # Actually, we'll use bounds for 0 <= q_ij <= 1
    # So we only need upper bounds if not handled by bounds parameter

    # IR: p_i <= v_i . q_i  =>  -v_i1*q_i1 - v_i2*q_i2 + p_i <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[q1_idx(i)] = -V[i, 0]
        row[q2_idx(i)] = -V[i, 1]
        row[p_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # IC: for all i, j: v_i . q_i - p_i >= v_i . q_j - p_j
    # => v_i1*(q_i1 - q_j1) + v_i2*(q_i2 - q_j2) + (p_j - p_i) >= 0
    # => -v_i1*q_i1 + v_i1*q_j1 - v_i2*q_i2 + v_i2*q_j2 - p_i + p_j <= 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[q1_idx(i)] = -V[i, 0]
            row[q1_idx(j)] = V[i, 0]
            row[q2_idx(i)] = -V[i, 1]
            row[q2_idx(j)] = V[i, 1]
            row[p_idx(i)] = -1.0
            row[p_idx(j)] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    # Bounds: 0 <= q_ij <= 1, p_i >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # q_i1
        bounds.append((0, 1))  # q_i2
    for i in range(n):
        bounds.append((0, None))  # p_i

    # Check time
    if time.time() > DEADLINE - 5:
        # Fallback: trivial mechanism
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}

    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback: trivial mechanism
            allocations = np.zeros((n, 2))
            payments = np.zeros(n)
            return {"allocations": allocations, "payments": payments}
    except Exception:
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}

    x = res.x

    # Extract allocations and payments
    allocations = np.zeros((n, 2))
    payments = np.zeros(n)
    for i in range(n):
        allocations[i, 0] = x[q1_idx(i)]
        allocations[i, 1] = x[q2_idx(i)]
        payments[i] = x[p_idx(i)]

    # Clip tiny negative values
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0.0)

    # Verify IC and IR with tolerance
    tol = 1e-8
    for i in range(n):
        utility_true = V[i, 0] * allocations[i, 0] + V[i, 1] * allocations[i, 1] - payments[i]
        if utility_true < -tol:
            # IR violated, fix by setting payment to value
            payments[i] = V[i, 0] * allocations[i, 0] + V[i, 1] * allocations[i, 1]

    # Check IC
    for i in range(n):
        u_i_true = V[i, 0] * allocations[i, 0] + V[i, 1] * allocations[i, 1] - payments[i]
        for j in range(n):
            u_i_report_j = V[i, 0] * allocations[j, 0] + V[i, 1] * allocations[j, 1] - payments[j]
            if u_i_true < u_i_report_j - tol:
                # IC violated - this shouldn't happen if LP was solved correctly
                # Adjust payment to restore IC
                pass  # The LP should have ensured this

    return {"allocations": allocations, "payments": payments}
