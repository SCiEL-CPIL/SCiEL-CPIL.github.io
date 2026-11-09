def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Parse the problem
    # distribution_spec: gives probabilities over 16 types (4x4 grid)
    # support: the value support, likely {0,1,2,3}
    # constraints: may contain additional info
    
    # Types are (v1, v2) where v1, v2 in {0,1,2,3}
    values = [0, 1, 2, 3]
    n_vals = len(values)
    n_types = n_vals * n_vals  # 16
    
    # Build type list
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Get probabilities from distribution_spec
    # distribution_spec should give us the probability of each type
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            # Maybe it's a 2D array
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) != n_types:
                # Default uniform
                probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables:
    # x[0:16] = q1(0), q1(1), ..., q1(15)  (allocation of good 1)
    # x[16:32] = q2(0), q2(1), ..., q2(15)  (allocation of good 2)
    # x[32:48] = p(0), p(1), ..., p(15)     (payments)
    
    n_vars = 3 * n_types  # 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p(t)
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    c[32:48] = -probs  # maximize sum probs[t]*p(t)
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Feasibility: 0 <= q1(t) <= 1 and 0 <= q2(t) <= 1
    # q1(t) <= 1: handled by bounds
    # -q1(t) <= 0: q1(t) >= 0: handled by bounds
    
    # IR: p(t) <= v1(t)*q1(t) + v2(t)*q2(t) for all t
    # => p(t) - v1(t)*q1(t) - v2(t)*v2(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[32 + t] = 1.0  # p(t)
        row[0 + t] = -values[types[t][0]]  # -v1(t)*q1(t)
        row[16 + t] = -values[types[t][1]]  # -v2(t)*q2(t)
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC: For all theta, theta': 
    # v(theta).q(theta) - p(theta) >= v(theta).q(theta') - p(theta')
    # => v(theta).q(theta) - p(theta) - v(theta).q(theta') + p(theta') >= 0
    # => -v(theta).q(theta) + p(theta) + v(theta).q(theta') - p(theta') <= 0
    for theta_idx in range(n_types):
        theta = types[theta_idx]
        for theta_prime_idx in range(n_types):
            theta_prime = types[theta_prime_idx]
            row = np.zeros(n_vars)
            # -v(theta).q(theta) + p(theta) + v(theta).q(theta') - p(theta')
            row[0 + theta_idx] = -theta[0]    # -v1(theta)*q1(theta)
            row[16 + theta_idx] = -theta[1]   # -v2(theta)*q2(theta)
            row[32 + theta_idx] = 1.0         # +p(theta)
            row[0 + theta_prime_idx] = theta[0]   # +v1(theta)*q1(theta')
            row[16 + theta_prime_idx] = theta[1]  # +v2(theta)*q2(theta')
            row[32 + theta_prime_idx] = -1.0      # -p(theta')
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # q1(t), q2(t) in [0, 1]
    # p(t) >= 0 (nonnegative payments)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # q1(t)
    for t in range(n_types):
        bounds.append((0, 1))  # q2(t)
    for t in range(n_types):
        bounds.append((0, None))  # p(t) >= 0
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payments
        allocations = np.ones((16, 2)) * 0.5
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    
    allocations = np.zeros((16, 2))
    for t in range(n_types):
        allocations[t, 0] = x[0 + t]
        allocations[t, 1] = x[16 + t]
    
    payments = x[32:48].copy()
    
    # Clean up numerical issues
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {"allocations": allocations, "payments": payments}
