"""Incentive-compatible finite auction design via direct LP with envelope-derived payments."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    
    # Parse support: list of (v1, v2) tuples, 16 types
    if isinstance(support, np.ndarray):
        support = [tuple(s) for s in support]
    types = sorted(support)
    n_types = len(types)
    
    # Build index mapping
    type_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution: probabilities for each type
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find any list-like key
            for k, v in distribution_spec.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == n_types:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # LP formulation:
    # Variables: x_1(v), x_2(v), p(v) for each type v
    # Total variables: 3 * n_types
    # Index: 0..n_types-1: x1(v), n_types..2*n_types-1: x2(v), 2*n_types..3*n_types-1: p(v)
    
    n_var = 3 * n_types
    
    def x1_idx(v): return type_idx[v]
    def x2_idx(v): return n_types + type_idx[v]
    def p_idx(v): return 2 * n_types + type_idx[v]
    
    # Objective: maximize sum_v pi(v) * (v1*x1(v) + v2*x2(v) - p(v))
    # linprog minimizes, so negate
    c = np.zeros(n_var)
    for i, v in enumerate(types):
        v1, v2 = v
        c[x1_idx(v)] = -probs[i] * v1
        c[x2_idx(v)] = -probs[i] * v2
        c[p_idx(v)] = probs[i]  # negative of -pi*p
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # Allocation bounds: 0 <= x_j(v) <= 1
    # Lower bounds handled by bounds parameter
    # Upper bounds: x_j(v) <= 1
    for v in types:
        A_rows.append(np.eye(n_var)[x1_idx(v)])
        b_rows.append(1.0)
        A_rows.append(np.eye(n_var)[x2_idx(v)])
        b_rows.append(1.0)
    
    # IR: v1*x1(v) + v2*x2(v) - p(v) >= 0
    # => -v1*x1(v) - v2*x2(v) + p(v) <= 0
    for v in types:
        v1, v2 = v
        row = np.zeros(n_var)
        row[x1_idx(v)] = -v1
        row[x2_idx(v)] = -v2
        row[p_idx(v)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC: v1*x1(v) + v2*x2(v) - p(v) >= v1*x1(v') + v2*x2(v') - p(v')
    # => v1*(x1(v)-x1(v')) + v2*(x2(v)-x2(v')) - (p(v)-p(v')) >= 0
    # => -v1*(x1(v)-x1(v')) - v2*(x2(v)-x2(v')) + (p(v)-p(v')) <= 0
    for v in types:
        v1, v2 = v
        for vp in types:
            if v == vp:
                continue
            row = np.zeros(n_var)
            row[x1_idx(v)] = -v1
            row[x2_idx(v)] = -v2
            row[p_idx(v)] = 1.0
            row[x1_idx(vp)] = v1
            row[x2_idx(vp)] = v2
            row[p_idx(vp)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: x_j(v) in [0,1], p(v) in [0, inf)
    bounds = []
    for v in types:
        bounds.append((0, 1))  # x1
        bounds.append((0, 1))  # x2
        bounds.append((0, None))  # p
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if time.time() > deadline - 5.0:
        # Time pressure: return a simple valid mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if res.success:
        x_sol = res.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for i, v in enumerate(types):
            allocations[i, 0] = x_sol[x1_idx(v)]
            allocations[i, 1] = x_sol[x2_idx(v)]
            payments[i] = x_sol[p_idx(v)]
    else:
        # Fallback: all-zero mechanism (trivially IC, IR, nonnegative payments)
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
    
    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify IC and IR with small tolerance; if violated, fall back to zero mechanism
    max_violation = 0.0
    for i, v in enumerate(types):
        v1, v2 = v
        util_truth = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
        # IR
        max_violation = max(max_violation, -util_truth)
        # IC
        for j, vp in enumerate(types):
            if i == j:
                continue
            util_deviate = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
            max_violation = max(max_violation, util_deviate - util_truth)
    
    if max_violation > 1e-6:
        # Try to repair by reducing allocations slightly or falling back
        # Simple fallback: zero mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
