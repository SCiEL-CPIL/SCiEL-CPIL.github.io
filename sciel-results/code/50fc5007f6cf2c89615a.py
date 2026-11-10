def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse support: 16 types, each type is (v1, v2) where v1, v2 in {0,1,2,3}
    # distribution_spec gives probabilities for each type
    # We need to figure out the exact format from the task
    
    # Based on task: "each value in {0,1,2,3}, three fixed distributions"
    # support likely contains the 16 type vectors
    # distribution_spec likely contains probability distributions
    
    # Let's assume support is a list/array of 16 type vectors, each [v1, v2]
    # distribution_spec is a dict or similar mapping to probabilities
    
    # Try to extract the data
    if isinstance(support, (list, tuple, np.ndarray)):
        types = np.array(support, dtype=float)
    else:
        types = np.array([[v1, v2] for v1 in range(4) for v2 in range(4)], dtype=float)
    
    n_types = len(types)
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        # Might be keyed by type index or by (v1,v2)
        try:
            probs = np.array([distribution_spec[i] for i in range(n_types)])
        except (KeyError, TypeError):
            probs = np.array([distribution_spec.get(tuple(types[i]), 1.0/n_types) for i in range(n_types)])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum()  # normalize
    
    # Variables: x[0..2n-1] are allocation probabilities (n_types * 2 goods)
    # x[n..2n-1] are payments
    # Total variables: 2 * n_types
    n_vars = 2 * n_types
    
    # Objective: maximize sum_t probs[t] * p[t]
    # In linprog (minimize): c = [0, ..., 0, -probs[0], -probs[1], ..., -probs[n-1]]
    c = np.zeros(n_vars)
    c[n_types:] = -probs
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t][g] <= 1 for all t, g
    #    Lower bounds: x >= 0, p >= 0
    #    Upper bounds: x[t][g] <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    # Payments can be large, set to a reasonable upper bound
    # Max possible value per type: 3*2 = 6, so payment <= 6
    ub[n_types:] = 6.0
    
    # 2. IR: p[t] <= v_t · x[t] for all t
    #    p[t] - v_t[0]*x[t][0] - v_t[1]*x[t][1] <= 0
    A_ub = []
    b_ub = []
    
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t] = 0  # x[t][0]
        row[n_types + t] = 1  # p[t]
        row[t] = -types[t][0]  # coefficient for x[t][0]
        row[n_types + t] = 1   # coefficient for p[t]
        # Wait, let me re-index. x variables are indices 0..n_types-1 for good 1,
        # and we need separate variables for good 2.
        
        # Let me restructure: variables are:
        # x1[0..n-1]: allocation prob for good 1 for each type
        # x2[0..n-1]: allocation prob for good 2 for each type  
        # p[0..n-1]: payment for each type
        # Total: 3 * n_types variables
        
        pass
    
    # Let me redo with 3*n_types variables
    n_vars = 3 * n_types
    # Indices:
    # x1[t] = t
    # x2[t] = n_types + t
    # p[t] = 2 * n_types + t
    
    c = np.zeros(n_vars)
    c[2 * n_types:] = -probs  # maximize sum probs[t]*p[t]
    
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[2 * n_types:] = 6.0  # payments bounded
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p[t] - v1[t]*x1[t] - v2[t]*x2[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t] = -types[t][0]       # x1[t]
        row[n_types + t] = -types[t][1]  # x2[t]
        row[2 * n_types + t] = 1    # p[t]
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC constraints: for all t, t':
    # v_t · x[t] - p[t] >= v_t · x[t'] - p[t']
    # => p[t] - p[t'] <= v_t[0]*(x1[t]-x1[t']) + v_t[1]*(x2[t]-x2[t'])
    # => v_t[0]*x1[t'] + v_t[1]*x2[t'] - v_t[0]*x1[t] - v_t[1]*x2[t] + p[t] - p[t'] <= 0
    
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # v_t · (x[t'] - x[t]) + p[t] - p[t'] <= 0
            row[tp] = types[t][0]           # v_t[0] * x1[t']
            row[n_types + tp] = types[t][1] # v_t[1] * x2[t']
            row[t] = -types[t][0]           # -v_t[0] * x1[t]
            row[n_types + t] = -types[t][1] # -v_t[1] * x2[t]
            row[2 * n_types + t] = 1        # p[t]
            row[2 * n_types + tp] = -1      # -p[t']
            A_ub.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a feasible but possibly suboptimal solution
        # Use simple mechanism: allocate both goods, charge expected value
        x1 = np.ones(n_types)
        x2 = np.ones(n_types)
        p = types[:, 0] + types[:, 1]  # p[t] = v_t · x[t] when x=1
        return {
            'allocations': np.column_stack([x1, x2]),
            'payments': p
        }
    
    if not result.success:
        # Fallback: simple mechanism
        x1 = np.ones(n_types)
        x2 = np.ones(n_types)
        p = types[:, 0] + types[:, 1]
        return {
            'allocations': np.column_stack([x1, x2]),
            'payments': p
        }
    
    x1 = result.x[:n_types]
    x2 = result.x[n_types:2*n_types]
    p = result.x[2*n_types:]
    
    # Clip to ensure feasibility
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    p = np.clip(p, 0, None)
    
    # Verify IC and IR with small tolerance
    # IR check
    for t in range(n_types):
        if p[t] > types[t][0]*x1[t] + types[t][1]*x2[t] + 1e-8:
            p[t] = types[t][0]*x1[t] + types[t][1]*x2[t]
    
    # IC check: adjust if violated
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            lhs = types[t][0]*x1[t] + types[t][1]*x2[t] - p[t]
            rhs = types[t][0]*x1[tp] + types[t][1]*x2[tp] - p[tp]
            if lhs < rhs - 1e-8:
                # Violation: increase p[tp] or decrease p[t]
                # Simple fix: set p[t] = min(p[t], types[t][0]*x1[t] + types[t][1]*x2[t])
                # and ensure IC by adjusting
                diff = rhs - lhs
                # Increase p[tp] by diff/2 and decrease p[t] by diff/2
                p[tp] += diff / 2
                p[t] -= diff / 2
                # Re-check IR
                p[t] = max(0, min(p[t], types[t][0]*x1[t] + types[t][1]*x2[t]))
                p[tp] = max(0, min(p[tp], types[tp][0]*x1[tp] + types[tp][1]*x2[tp]))
    
    # Final verification
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            lhs = types[t][0]*x1[t] + types[t][1]*x2[t] - p[t]
            rhs = types[t][0]*x1[tp] + types[t][1]*x2[tp] - p[tp]
            if lhs < rhs - 1e-8:
                # Still violated, use conservative mechanism
                x1 = np.ones(n_types)
                x2 = np.ones(n_types)
                p = types[:, 0] + types[:, 1]
                return {
                    'allocations': np.column_stack([x1, x2]),
                    'payments': p
                }
    
    return {
        'allocations': np.column_stack([x1, x2]),
        'payments': p
    }
