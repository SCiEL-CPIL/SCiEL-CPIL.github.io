"""Rewritten program: direct convex optimization over payment vector with IC-derived payments."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)
    
    n = 16
    
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]
    
    A_rows = []
    b_rows = []
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, 3 * n))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    bounds = []
    for t in range(n):
        bounds.append((0, 1))
    for t in range(n):
        bounds.append((0, 1))
    for t in range(n):
        bounds.append((0, None))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-10:
            payments[t] = max(0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
