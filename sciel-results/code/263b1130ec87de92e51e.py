import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Determine the types
    if support is not None and len(support) == 16:
        types = [tuple(t) for t in support]
    else:
        types = [(v1, v2) for v1 in [0,1,2,3] for v2 in [0,1,2,3]]
    
    n_types = len(types)
    
    # Get probabilities
    probs = np.ones(n_types) / n_types
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float).flatten()
            if len(arr) == n_types:
                probs = arr
            else:
                # Try to match
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    
    # Normalize
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # LP setup
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    def x_idx(θ, i):
        return θ * 2 + i
    
    def p_idx(θ):
        return n_x + θ
    
    # Objective: min -Σ P(θ) p_θ
    c = np.zeros(n_vars)
    for θ in range(n_types):
        c[p_idx(θ)] = -probs[θ]
    
    A_rows = []
    b_ub = []
    
    # IC constraints
    for θ in range(n_types):
        for θ_prime in range(n_types):
            if θ == θ_prime:
                continue
            row = np.zeros(n_vars)
            row[p_idx(θ)] = 1.0
            row[p_idx(θ_prime)] = -1.0
            for i in range(2):
                row[x_idx(θ, i)] = -float(types[θ][i])
                row[x_idx(θ_prime, i)] = float(types[θ_prime][i])
            A_rows.append(row)
            b_ub.append(0.0)
    
    # IR constraints
    for θ in range(n_types):
        row = np.zeros(n_vars)
        row[p_idx(θ)] = 1.0
        for i in range(2):
            row[x_idx(θ, i)] = -float(types[θ][i])
        A_rows.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub) if b_ub else np.zeros(0)
    
    bounds = []
    for _ in range(n_x):
        bounds.append((0, 1))
    for _ in range(n_p):
        bounds.append((0, None))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
    
    x_vals = result.x[:n_x].reshape(n_types, 2)
    p_vals = result.x[n_x:]
    
    x_vals = np.clip(x_vals, 0.0, 1.0)
    p_vals = np.clip(p_vals, 0.0, None)
    
    return {"allocations": x_vals, "payments": p_vals}
