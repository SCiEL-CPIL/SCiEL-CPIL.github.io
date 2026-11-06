import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Determine types
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n = len(types)  # 16
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float).flatten()
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float).flatten()
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float).flatten()
        else:
            probs = np.ones(n) / n
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
    else:
        probs = np.ones(n) / n
    
    if len(probs) != n:
        # Try to infer from support
        if support is not None:
            probs = np.ones(len(support)) / len(support)
            n = len(support)
            # Rebuild types from support
            types = list(support)
        else:
            probs = np.ones(n) / n
    
    # Normalize
    probs = probs / probs.sum()
    
    # LP formulation
    n_vars = 3 * n
    
    # Objective: maximize sum_v probs[v] * p[v]
    c = np.zeros(n_vars)
    c[2*n:] = -probs
    
    # Bounds
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, None)] * n
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # IR: p(v) <= v1*q1(v) + v2*q2(v)
    for i in range(n):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[n + i] = -v2
        row[2*n + i] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # IC: for all v (true), v' (reported)
    for i in range(n):
        v1, v2 = types[i]
        for j in range(n):
            v1p, v2p = types[j]
            row = np.zeros(n_vars)
            row[j] = v1p
            row[n + j] = v2p
            row[2*n + j] = -1
            row[i] = -v1
            row[n + i] = -v2
            row[2*n + i] = 1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                           'primal_feasibility_tolerance': 1e-12})
    
    if not res.success:
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    x = res.x
    q1 = x[0:n]
    q2 = x[n:2*n]
    p = x[2*n:3*n]
    
    # Clip to [0,1] for safety
    q1 = np.clip(q1, 0, 1)
    q2 = np.clip(q2, 0, 1)
    p = np.clip(p, 0, None)
    
    allocations = np.column_stack([q1, q2])
    payments = p
    
    return {'allocations': allocations, 'payments': payments}
