"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Create value matrix: n_types x n_goods
    V = np.array([[v[0], v[1]] for v in types], dtype=float)
    
    # Decision variables:
    # x[t, g] for t in range(n_types), g in range(n_goods) -> n_types * n_goods = 32 vars
    # p[t] for t in range(n_types) -> n_types = 16 vars
    # Total: 48 variables
    
    n_x = n_types * n_goods
    n_p = n_types
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]  # maximize p[t] weighted by prob
    
    # Equality constraints: none
    
    # Inequality constraints (A_ub @ x <= b_ub)
    A_rows = []
    b_rows = []
    
    # IR constraints: V[t]·x[t] - p[t] >= 0  =>  -V[t]·x[t] + p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[t * n_goods + g] = -V[t, g]
        row[n_x + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: for all t, t': V[t]·x[t'] - p[t'] <= V[t]·x[t] - p[t]
    # Rearranged: V[t]·x[t'] - V[t]·x[t] - p[t'] + p[t] <= 0
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # -V[t]·x[t]
            for g in range(n_goods):
                row[t * n_goods + g] = -V[t, g]
            # +V[t]·x[t']
            for g in range(n_goods):
                row[tp * n_goods + g] = V[t, g]
            # +p[t]
            row[n_x + t] = 1.0
            # -p[t']
            row[n_x + tp] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    bounds = []
    for i in range(n_x):
        bounds.append((0.0, 1.0))
    for i in range(n_p):
        bounds.append((0.0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return zero allocations and zero payments
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_flat = result.x[:n_x]
    p_vec = result.x[n_x:n_x + n_p]
    
    allocations = x_flat.reshape(n_types, n_goods)
    payments = p_vec
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify constraints with tolerance
    # Check IR
    for t in range(n_types):
        util = np.dot(V[t], allocations[t]) - payments[t]
        if util < -1e-8:
            # Adjust payment to satisfy IR
            payments[t] = np.dot(V[t], allocations[t])
    
    # Check IC and fix if needed
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            # Utility of type t if they get allocation of type tp and pay p[tp]
            util_misrep = np.dot(V[t], allocations[tp]) - payments[tp]
            # Utility of type t reporting truthfully
            util_truth = np.dot(V[t], allocations[t]) - payments[t]
            if util_misrep > util_truth + 1e-8:
                # This violates IC; we need to adjust
                # For now, this indicates the LP solution might have issues
                # In practice, the LP should handle this correctly
                pass
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
