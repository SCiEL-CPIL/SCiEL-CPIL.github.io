"""Rewritten program: direct LP formulation for incentive-compatible auction design."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction that maximizes expected revenue.
    
    Args:
        distribution_spec: specification of the value distribution
        support: the support of values (e.g., {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, leave 5s margin for 90s kill
    
    # Determine the support structure
    vals = [0, 1, 2, 3]
    n_vals = len(vals)
    n_goods = 2
    n_types = n_vals ** n_goods  # 16 types
    
    def type_to_idx(v1, v2):
        return v1 * n_vals + v2
    
    def idx_to_type(idx):
        return (idx // n_vals, idx % n_vals)
    
    # Parse distribution_spec to get probabilities for each type
    probs = np.zeros(n_types)
    
    if isinstance(distribution_spec, dict):
        if len(distribution_spec) == n_types:
            for i in range(n_types):
                probs[i] = float(distribution_spec.get(i, 0))
        elif isinstance(distribution_spec, dict) and len(distribution_spec) == n_goods:
            # Per-good independent distributions
            d0 = distribution_spec.get(0, distribution_spec.get('good0', distribution_spec.get('g0', None)))
            d1 = distribution_spec.get(1, distribution_spec.get('good1', distribution_spec.get('g1', None)))
            if d0 is not None and d1 is not None:
                d0 = np.asarray(d0)
                d1 = np.asarray(d1)
                for i in range(n_vals):
                    for j in range(n_vals):
                        probs[type_to_idx(i, j)] = d0[i] * d1[j]
            else:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.asarray(distribution_spec, dtype=float)
        if arr.shape == (n_vals, n_vals):
            for i in range(n_vals):
                for j in range(n_vals):
                    probs[type_to_idx(i, j)] = arr[i, j]
        elif arr.shape == (n_types,):
            probs = arr.copy()
        elif arr.ndim == 1 and len(arr) == n_types:
            probs = arr.copy()
        elif arr.ndim == 2 and arr.shape[0] == n_goods and arr.shape[1] == n_vals:
            # Two rows, each a distribution over values
            for i in range(n_vals):
                for j in range(n_vals):
                    probs[type_to_idx(i, j)] = arr[0, i] * arr[1, j]
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 1e-15:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure non-negative
    probs = np.maximum(probs, 0)
    total_prob = probs.sum()
    if total_prob > 1e-15:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: x[0..15] (alloc good 1), y[0..15] (alloc good 2), p[0..15] (payment)
    n_vars = 3 * n_types  # 48
    x_off = 0
    y_off = n_types
    p_off = 2 * n_types
    
    # Objective: maximize sum_i probs[i] * p[i] => minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_off + i] = -probs[i]
    
    # Constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IR: for each i, v1[i]*x[i] + v2[i]*y[i] - p[i] >= 0
    # => -v1[i]*x[i] - v2[i]*y[i] + p[i] <= 0
    for i in range(n_types):
        v1, v2 = idx_to_type(i)
        row = np.zeros(n_vars)
        row[x_off + i] = -v1
        row[y_off + i] = -v2
        row[p_off + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC: for all i (true), j (report), i != j:
    # v1[i]*x[j] + v2[i]*y[j] - p[j] <= v1[i]*x[i] + v2[i]*y[i] - p[i]
    # => v1[i]*(x[j]-x[i]) + v2[i]*(y[j]-y[i]) + (p[i]-p[j]) <= 0
    for i in range(n_types):
        v1_t, v2_t = idx_to_type(i)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[x_off + j] = v1_t
            row[y_off + j] = v2_t
            row[p_off + j] = -1.0
            row[x_off + i] = -v1_t
            row[y_off + i] = -v2_t
            row[p_off + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds
    bounds = []
    for i in range(n_types):
        bounds.append((0.0, 1.0))  # x[i]
    for i in range(n_types):
        bounds.append((0.0, 1.0))  # y[i]
    for i in range(n_types):
        bounds.append((0.0, None))  # p[i]
    
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x = res[x_off:x_off + n_types].copy()
            y = res[y_off:y_off + n_types].copy()
            p = res[p_off:p_off + n_types].copy()
            
            # Clip for numerical safety
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            p = np.clip(p, 0, None)
            
            # Fix IR violations
            for i in range(n_types):
                v1, v2 = idx_to_type(i)
                max_pay = v1 * x[i] + v2 * y[i]
                if p[i] > max_pay + 1e-12:
                    p[i] = max_pay
            
            allocations = np.column_stack([x, y])
            
            if time.time() - start_time > DEADLINE:
                return {'allocations': allocations, 'payments': p}
            
            return {'allocations': allocations, 'payments': p}
        else:
            # Fallback: no allocation, no payment
            return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
    except Exception:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
