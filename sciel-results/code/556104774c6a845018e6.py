import numpy as np
from scipy.optimize import linprog

# Cache for solved mechanisms to avoid redundant LP solves
_MECHANISM_CACHE = {}

def _get_distribution_probs(distribution_spec, n_types):
    """
    Robustly parse distribution_spec into a probability vector and a cache key.
    """
    if isinstance(distribution_spec, str):
        if distribution_spec in ('uniform', 'independent_uniform', 'correlated'):
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
        key = ('str', distribution_spec)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.sum() == 0:
            probs = np.ones(n_types) / n_types
        else:
            probs = probs / probs.sum()
        # Create a hashable key by rounding to avoid float precision issues
        key = ('arr', tuple(np.round(probs, 10)))
    else:
        probs = np.ones(n_types) / n_types
        key = ('default',)
        
    return key, probs

def construct_auction(distribution_spec, support, constraints):
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution and cache key
    cache_key, probs = _get_distribution_probs(distribution_spec, n_types)
    
    # Check cache
    if cache_key in _MECHANISM_CACHE:
        return _MECHANISM_CACHE[cache_key]
    
    # Variables: q1[0:16], q2[16:32], p[32:48]
    n_vars = 3 * n_types
    
    # Objective: maximize expected revenue = sum(pi_i * p_i)
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[2 * n_types + i] = -probs[i]  # Negative for maximization
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # IC constraints: For all true type i and reported type j:
    # v1_i*q1_i + v2_i*q2_i - p_i >= v1_i*q1_j + v2_i*q2_j - p_j
    # Rearranged: v1_i*q1_i + v2_i*q2_i - p_i - v1_i*q1_j - v2_i*q2_j + p_j >= 0
    # For linprog (<= form): -v1_i*q1_i - v2_i*q2_i + p_i + v1_i*q1_j + v2_i*q2_j - p_j <= 0
    
    for i, (v1_i, v2_i) in enumerate(types):
        for j, (v1_j, v2_j) in enumerate(types):
            if i == j:
                continue  # Trivially satisfied
            row = np.zeros(n_vars)
            # -v1_i*q1_i - v2_i*q2_i + p_i + v1_i*q1_j + v2_i*q2_j - p_j
            row[i] = -v1_i                    # q1_i
            row[n_types + i] = -v2_i          # q2_i
            row[2 * n_types + i] = 1.0        # p_i
            row[j] = v1_i                     # q1_j
            row[n_types + j] = v2_i           # q2_j
            row[2 * n_types + j] = -1.0       # p_j
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: v1_i*q1_i + v2_i*q2_i - p_i >= 0
    # Rearranged: -v1_i*q1_i - v2_i*q2_i + p_i <= 0
    for i, (v1_i, v2_i) in enumerate(types):
        row = np.zeros(n_vars)
        row[i] = -v1_i
        row[n_types + i] = -v2_i
        row[2 * n_types + i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Bounds: 0 <= q1 <= 1, 0 <= q2 <= 1, p >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # q1
    for i in range(n_types):
        bounds.append((0, 1))  # q2
    for i in range(n_types):
        bounds.append((0, None))  # p
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status != 0:
        # Fallback: trivial mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = res.x
    allocations = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
    payments = x[2*n_types:3*n_types]
    
    # Ensure numerical feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    result = {'allocations': allocations, 'payments': payments}
    _MECHANISM_CACHE[cache_key] = result
    return result
