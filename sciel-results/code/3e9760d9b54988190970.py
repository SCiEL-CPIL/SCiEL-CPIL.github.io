import numpy as np
from itertools import product
from scipy.optimize import linprog
import time

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
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill

    # Fixed support as per task description
    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    type_to_vals = {i: t for i, t in enumerate(types)}

    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'values' in distribution_spec and 'probs' in distribution_spec:
                vals = np.array(distribution_spec['values'])
                ps = np.array(distribution_spec['probs'])
                if len(vals) == 16 and len(ps) == 16:
                    probs = ps
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    n = 16
    
    # Fallback mechanism: Allocate only to the highest type (3,3)
    # This is always valid and yields non-zero revenue if pi(3,3) > 0.
    def get_fallback():
        allocs = np.zeros((16, 2))
        # Find index of (3,3)
        idx_33 = types.index((3,3))
        allocs[idx_33] = [1, 1]
        # Payment for (3,3) must be >= max(V(t)) for t != (3,3)
        # V(t) = v1 + v2
        max_v_other = 0
        for t in range(16):
            if t == idx_33: continue
            v1, v2 = type_to_vals[t]
            max_v_other = max(max_v_other, v1 + v2)
        pays = np.zeros(16)
        pays[idx_33] = max_v_other
        return {'allocations': allocs, 'payments': pays}

    # LP Formulation
    # Variables: x1[16], x2[16], p[16]
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_rows = []
    b_rows = []
    
    # IC Constraints
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    # IR Constraints
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, None))

    options = {
        'presolve': True,
        'dual_feasibility_tolerance': 1e-9,
        'primal_feasibility_tolerance': 1e-9,
        'dual_gap_tolerance': 1e-9
    }
    
    result = None
    try:
        if time.time() - start_time < timeout:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        result = None

    if result is None or not result.success:
        return get_fallback()

    x = result.x
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Polish payments
    c_polish = -probs
    A_polish = []
    b_polish = []
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2: continue
            v1_t2, v2_t2 = type_to_vals[t2]
            v_t_x_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            v_t2_x_t2 = v1_t2 * allocations[t2, 0] + v2_t2 * allocations[t2, 1]
            row = np.zeros(n)
            row[t] = 1
            row[t2] = -1
            A_polish.append(row)
            b_polish.append(v_t_x_t - v_t2_x_t2)
            
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        v_t_x_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        row = np.zeros(n)
        row[t] = 1
        A_polish.append(row)
        b_polish.append(v_t_x_t)
        
    A_polish = np.array(A_polish)
    b_polish = np.array(b_polish)
    bounds_polish = [(0, None)] * n
    
    try:
        result_polish = linprog(c_polish, A_ub=A_polish, b_ub=b_polish, bounds=bounds_polish, method='highs', options=options)
        if result_polish.success:
            payments = result_polish.x
            payments = np.maximum(payments, 0)
    except Exception:
        pass
        
    # Final validation
    valid = True
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t + 1e-9:
            valid = False
            break
        for t2 in range(n):
            if t == t2: continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = payments[t] - payments[t2]
            rhs = v1_t * (allocations[t, 0] - allocations[t2, 0]) + v2_t * (allocations[t, 1] - allocations[t2, 1])
            if lhs > rhs + 1e-9:
                valid = False
                break
        if not valid: break
        
    if not valid:
        return get_fallback()
        
    return {'allocations': allocations, 'payments': payments}
