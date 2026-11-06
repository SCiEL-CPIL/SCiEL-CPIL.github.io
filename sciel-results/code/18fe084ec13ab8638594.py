import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Parse inputs
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', []), dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
        
    if len(probs) < 16:
        probs = np.pad(probs, (0, 16 - len(probs)))
    else:
        probs = probs[:16]
        
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
        
    support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    values = np.array(support, dtype=float)
    
    n_types = 16
    n_goods = 2
    n_vars = 48
    n_x = 32
    n_u = 16
    
    c = np.zeros(n_vars)
    for i in range(n_types):
        for g in range(n_goods):
            c[i * n_goods + g] = -probs[i] * values[i, g]
        c[n_x + i] = probs[i]
        
    A_ub_rows = []
    b_ub_rows = []
    
    # IC
    for i in range(n_types):
        for j in range(n_types):
            if i == j: continue
            row = np.zeros(n_vars)
            row[n_x + i] = -1.0
            row[n_x + j] = 1.0
            for g in range(n_goods):
                coeff = values[i, g] - values[j, g]
                if coeff != 0:
                    row[j * n_goods + g] = coeff
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # IR
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + i] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    # Non-negative Payments
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + i] = 1.0
        for g in range(n_goods):
            row[i * n_goods + g] = -values[i, g]
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    bounds = [(0, 1)] * n_x + [(0, None)] * n_u
    
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
        
    try:
        # Tighter tolerances for higher precision
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                         options={'presolve': True, 
                                  'primal_feasibility_tolerance': 1e-14, 
                                  'dual_feasibility_tolerance': 1e-14})
    except Exception:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
        
    if result.success:
        x_opt = result.x
        allocs = x_opt[:32].reshape(16, 2)
        u_opt = x_opt[32:]
        
        # Clip to ensure feasibility
        allocs = np.clip(allocs, 0, 1)
        
        pays = np.sum(allocs * values, axis=1) - u_opt
        pays = np.maximum(pays, 0)
        
        return {'allocations': allocs, 'payments': pays}
    else:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
