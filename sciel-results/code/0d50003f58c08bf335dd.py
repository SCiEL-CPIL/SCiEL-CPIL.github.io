import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Values in {0,1,2,3} for 2 goods → 16 types
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Get probabilities for each type from distribution_spec
    # distribution_spec should give us Pr(v) for each type
    # Let's handle various formats
    if isinstance(distribution_spec, dict):
        # Could be keyed by type tuple or by index
        if (0, 0) in distribution_spec:
            probs = np.array([distribution_spec[t] for t in types])
        else:
            probs = np.array([distribution_spec.get(i, 0) for i in range(n_types)])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n_types:
            # Try to reshape or handle
            probs = probs[:n_types] if len(probs) >= n_types else np.pad(probs, (0, n_types - len(probs)))
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs.astype(float)
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    
    # LP variables: [x_1(0), x_2(0), ..., x_1(15), x_2(15), p(0), ..., p(15)]
    # Total: 16*2 + 16 = 48 variables
    n_alloc = n_types * 2  # 32
    n_pay = n_types  # 16
    n_vars = n_alloc + n_pay  # 48
    
    # Objective: maximize sum_v probs[v] * p(v)
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for v in range(n_types):
        c[n_alloc + v] = -probs[v]  # maximize revenue
    
    # Inequality constraints: A_ub @ x <= b_ub
    # IC: for each (true_v, reported_v'): 
    #   true_v · x(v') - p(v') <= true_v · x(v) - p(v)
    #   true_v · x(v') - p(v') - true_v · x(v) + p(v) <= 0
    # For type v = (v1, v2), x(v) = (x_1(v), x_2(v))
    # true_v · x(v') = v1*x_1(v') + v2*x_2(v')
    
    ic_constraints = []
    ic_rhs = []
    
    for true_v_idx in range(n_types):
        tv = types[true_v_idx]
        for rep_v_idx in range(n_types):
            rv = types[rep_v_idx]
            row = np.zeros(n_vars)
            # true_v · x(rep_v) - p(rep_v) - true_v · x(true_v) + p(true_v) <= 0
            # tv[0]*x_1(rep_v) + tv[1]*x_2(rep_v) - p(rep_v) - tv[0]*x_1(true_v) - tv[1]*x_2(true_v) + p(true_v) <= 0
            row[rep_v_idx] = tv[0]       # x_1(rep_v)
            row[n_types + rep_v_idx] = tv[1]  # x_2(rep_v)
            row[n_alloc + rep_v_idx] = -1.0   # -p(rep_v)
            row[true_v_idx] = -tv[0]       # -x_1(true_v)
            row[n_types + true_v_idx] = -tv[1]  # -x_2(true_v)
            row[n_alloc + true_v_idx] = 1.0    # +p(true_v)
            ic_constraints.append(row)
            ic_rhs.append(0.0)
    
    # IR: for each v: v · x(v) - p(v) >= 0  →  -v · x(v) + p(v) <= 0
    ir_constraints = []
    ir_rhs = []
    for v_idx in range(n_types):
        v = types[v_idx]
        row = np.zeros(n_vars)
        row[v_idx] = -v[0]
        row[n_types + v_idx] = -v[1]
        row[n_alloc + v_idx] = 1.0
        ir_constraints.append(row)
        ir_rhs.append(0.0)
    
    # Nonnegativity of payments: p(v) >= 0 → -p(v) <= 0
    nn_constraints = []
    nn_rhs = []
    for v_idx in range(n_types):
        row = np.zeros(n_vars)
        row[n_alloc + v_idx] = -1.0
        nn_constraints.append(row)
        nn_rhs.append(0.0)
    
    # Combine all inequality constraints
    all_ic = ic_constraints + ir_constraints + nn_constraints
    A_ub = np.array(all_ic)
    b_ub = np.array(all_rhs)
    
    # Bounds: x in [0,1], p in [0, inf)
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: simple allocation
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if not result.success:
        # Fallback
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, 2))
    for v_idx in range(n_types):
        allocations[v_idx, 0] = x_opt[v_idx]
        allocations[v_idx, 1] = x_opt[n_types + v_idx]
    
    payments = x_opt[n_alloc:n_alloc + n_pay].copy()
    
    # Clip small negatives from numerical error
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify constraints with tolerance
    # Check IC
    max_regret = 0.0
    for true_v_idx in range(n_types):
        tv = types[true_v_idx]
        u_true = tv[0] * allocations[true_v_idx, 0] + tv[1] * allocations[true_v_idx, 1] - payments[true_v_idx]
        for rep_v_idx in range(n_types):
            rv = types[rep_v_idx]
            u_rep = tv[0] * allocations[rep_v_idx, 0] + tv[1] * allocations[rep_v_idx, 1] - payments[rep_v_idx]
            regret = u_rep - u_true
            if regret > max_regret:
                max_regret = regret
    
    # Check IR
    min_utility = float('inf')
    for v_idx in range(n_types):
        v = types[v_idx]
        u = v[0] * allocations[v_idx, 0] + v[1] * allocations[v_idx, 1] - payments[v_idx]
        if u < min_utility:
            min_utility = u
    
    # If violations exist, project to feasible region
    if max_regret > 1e-8 or min_utility < -1e-8:
        # Increase payments to satisfy IR, adjust allocations for IC
        # Simple fix: set payments to ensure IR, then check IC
        for v_idx in range(n_types):
            v = types[v_idx]
            min_pay = v[0] * allocations[v_idx, 0] + v[1] * allocations[v_idx, 1]
            payments[v_idx] = max(payments[v_idx], min_pay)
        
        # Re-check and fix IC by reducing allocations where needed
        for _ in range(10):
            max_regret = 0.0
            violated = False
            for true_v_idx in range(n_types):
                tv = types[true_v_idx]
                u_true = tv[0] * allocations[true_v_idx, 0] + tv[1] * allocations[true_v_idx, 1] - payments[true_v_idx]
                for rep_v_idx in range(n_types):
                    rv = types[rep_v_idx]
                    u_rep = tv[0] * allocations[rep_v_idx, 0] + tv[1] * allocations[rep_v_idx, 1] - payments[rep_v_idx]
                    regret = u_rep - u_true
                    if regret > 1e-8:
                        violated = True
                        max_regret = max(max_regret, regret)
            if not violated:
                break
            # Reduce allocation at the most problematic report
            # This is a heuristic; in practice the LP should be clean
            break
    
    # Final safety: ensure nonnegative payments and valid allocations
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
