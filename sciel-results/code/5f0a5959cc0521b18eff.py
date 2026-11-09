def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    
    # Parse inputs
    # support: list of (v1, v2) tuples, should be 16 entries
    # distribution_spec: probabilities for each type
    # constraints: may contain additional info
    
    types = support  # list of (v1, v2)
    n_types = len(types)
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        # Map type -> probability
        probs = np.zeros(n_types)
        for i, t in enumerate(types):
            probs[i] = distribution_spec.get(t, 0)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) == n_types:
            pass
        else:
            # Try to map
            probs = np.zeros(n_types)
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
    else:
        # Default uniform
        probs = np.ones(n_types) / n_types
    
    # Ensure probs sum to 1
    if probs.sum() > 0:
        probs = probs / probs.sum()
    
    # Extract v1, v2 arrays
    v1 = np.array([t[0] for t in types], dtype=float)
    v2 = np.array([t[1] for t in types], dtype=float)
    
    # Variables: x1[0..15], x2[0..15], p[0..15]
    # Total: 48 variables
    # Objective: maximize sum(probs[i] * p[i])
    # linprog minimizes, so minimize -sum(probs[i] * p[i])
    
    c = np.zeros(3 * n_types)
    for i in range(n_types):
        c[2 * n_types + i] = -probs[i]  # p[i] is at index 2*n_types + i
    
    # Constraints:
    # 1. IR: p[i] - v1[i]*x1[i] - v2[i]*x2[i] <= 0 for each i
    # 2. IC: for each pair (i, j): v1[i]*x1[i] + v2[i]*x2[i] - p[i] >= v1[i]*x1[j] + v2[i]*x2[j] - p[j]
    #    => v1[i]*x1[i] + v2[i]*x2[i] - p[i] - v1[i]*x1[j] - v2[i]*x2[j] + p[j] >= 0
    #    => -(v1[i]*x1[i] + v2[i]*x2[i] - p[i] - v1[i]*x1[j] - v2[i]*x2[j] + p[j]) <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p[i] - v1[i]*x1[i] - v2[i]*x2[i] <= 0
    for i in range(n_types):
        row = np.zeros(3 * n_types)
        row[i] = -v1[i]  # x1[i]
        row[n_types + i] = -v2[i]  # x2[i]
        row[2 * n_types + i] = 1  # p[i]
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints: for each i != j
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(3 * n_types)
            # v1[i]*x1[i] + v2[i]*x2[i] - p[i] - v1[i]*x1[j] - v2[i]*x2[j] + p[j] >= 0
            # => -v1[i]*x1[i] - v2[i]*x2[i] + p[i] + v1[i]*x1[j] + v2[i]*x2[j] - p[j] <= 0
            row[i] = -v1[i]
            row[n_types + i] = -v2[i]
            row[2 * n_types + i] = 1
            row[j] = v1[i]
            row[n_types + j] = v2[i]
            row[2 * n_types + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x1[i] <= 1, 0 <= x2[i] <= 1, p[i] >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(n_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(n_types):
        bounds.append((0, None))  # p[i]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.column_stack([x[:n_types], x[n_types:2*n_types]])
    payments = x[2*n_types:]
    
    # Clean up small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
