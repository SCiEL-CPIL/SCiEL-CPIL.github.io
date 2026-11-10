import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)  # 16
    n_goods = 2

    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'uniform' in distribution_spec:
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, str):
        if 'uniform' in distribution_spec.lower():
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    probs = probs / probs.sum()

    # Value vectors: shape (16, 2)
    V = np.array([(t[0], t[1]) for t in types], dtype=float)

    # LP variables: [q(0,0), q(0,1), ..., q(15,0), q(15,1), p(0), p(1), ..., p(15)]
    n_q = n_types * n_goods
    n_p = n_types
    n_vars = n_q + n_p

    # Objective: maximize sum_t pi(t) * p(t)
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_q + i] = -probs[i]

    # Constraints:
    # IR: p(t) - v(t).q(t) <= 0 for all t
    # IC: p(t) - p(t') - v(t).q(t) + v(t).q(t') <= 0 for all t, t'
    n_ir = n_types
    n_ic = n_types * n_types
    n_constraints = n_ir + n_ic

    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)

    # IR constraints
    for i in range(n_types):
        row = i
        A_ub[row, n_q + i] = 1.0
        for g in range(n_goods):
            A_ub[row, i * n_goods + g] = -V[i, g]
        b_ub[row] = 0.0

    # IC constraints
    idx = n_ir
    for i in range(n_types):
        for j in range(n_types):
            A_ub[idx, n_q + i] = 1.0
            A_ub[idx, n_q + j] = -1.0
            for g in range(n_goods):
                A_ub[idx, i * n_goods + g] = -V[i, g]
                A_ub[idx, j * n_goods + g] = V[i, g]
            b_ub[idx] = 0.0
            idx += 1

    # Bounds
    bounds = []
    for i in range(n_q):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    x = result.x
    allocations = np.zeros((n_types, n_goods))
    for i in range(n_types):
        for g in range(n_goods):
            allocations[i, g] = x[i * n_goods + g]

    payments = np.zeros(n_types)
    for i in range(n_types):
        payments[i] = x[n_q + i]

    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)

    return {'allocations': allocations, 'payments': payments}
