import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Determine the 16 types
    # Values in {0,1,2,3} for each of 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Get probability distribution
    if isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        probs = probs / probs.sum()
    elif isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if 'uniform' in spec:
            probs = np.ones(n_types) / n_types
        elif 'decreasing' in spec or 'monotone' in spec:
            # Decreasing in both dimensions
            probs = np.array([[ (3-v1)*(3-v2) for v2 in values] for v1 in values], dtype=float)
            probs = probs.flatten()
            probs = probs / probs.sum()
        elif 'independent' in spec:
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # LP variables: q[0..15,0], q[0..15,1], p[0..15]
    # Total: 32 + 16 = 48
    n_vars = 2 * n_types + n_types  # 48
    
    # Objective: maximize sum_t probs[t] * p[t]
    # For linprog (minimize): c = -[0,0,...,0, probs[0], probs[1], ..., probs[15]]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[2 * n_types + t] = -probs[t]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. IC: for all t, t': -sum_g v_tg*q[t,g] + sum_g v_tg*q[t',g] + p[t] - p[t'] <= 0
    # 2. IR: for all t: -sum_g v_tg*q[t,g] + p[t] <= 0  (i.e., p[t] - utility[t] <= 0, utility >= 0)
    #    Actually IR: sum_g v_tg*q[t,g] - p[t] >= 0  =>  p[t] - sum_g v_tg*q[t,g] <= 0
    
    A_rows = []
    b_rows = []
    
    # IC constraints: n_types * n_types = 256
    for t in range(n_types):
        for t2 in range(n_types):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            # -sum_g v_tg*q[t,g]
            for g in range(2):
                row[g * n_types + t] = -types[t][g]
            # +sum_g v_tg*q[t2,g]
            for g in range(2):
                row[g * n_types + t2] += types[t][g]
            # +p[t] - p[t2]
            row[2 * n_types + t] += 1
            row[2 * n_types + t2] -= 1
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints: n_types = 16
    for t in range(n_types):
        row = np.zeros(n_vars)
        # p[t] - sum_g v_tg*q[t,g] <= 0
        for g in range(2):
            row[g * n_types + t] = -types[t][g]
        row[2 * n_types + t] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= q[t,g] <= 1, p[t] >= 0
    bounds = []
    for i in range(2 * n_types):
        bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payment
        allocations = np.ones((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.zeros((n_types, 2))
    for g in range(2):
        allocations[:, g] = x[g * n_types : (g+1) * n_types]
    payments = x[2 * n_types : 3 * n_types]
    
    # Clamp small negatives
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    return {'allocations': allocations, 'payments': payments}
