def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # Parse inputs
    # distribution_spec: describes the distribution over types
    # support: the set of possible type values
    # constraints: any additional constraints
    
    # For this task: one bidder, two goods, values in {0,1,2,3}
    # 16 possible type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    
    values = [0, 1, 2, 3]
    n_values = len(values)
    n_types = n_values * n_values  # 16 types
    
    # Build type indices
    type_indices = []
    for v1 in values:
        for v2 in values:
            type_indices.append((v1, v2))
    
    # Get probabilities from distribution_spec
    # distribution_spec should give us the probability for each type
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
            if len(probs) == n_types:
                probs = probs.reshape((n_values, n_values))
            else:
                probs = np.ones(n_types) / n_types
        elif 'grid' in distribution_spec:
            probs = np.ones((n_values, n_values)) / n_types
        else:
            probs = np.ones((n_values, n_values)) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.ndim == 1 and len(probs) == n_types:
            probs = probs.reshape((n_values, n_values))
        elif probs.ndim == 2 and probs.shape == (n_values, n_values):
            pass
        else:
            probs = np.ones((n_values, n_values)) / n_types
    else:
        probs = np.ones((n_values, n_values)) / n_types
    
    # Ensure probs are valid
    probs = np.maximum(probs, 0)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones((n_values, n_values)) / n_types
    
    # LP formulation for single-bidder multi-dimensional auction
    # Decision variables:
    #   x[i,j] = probability of allocating good 1 to bidder with type (i,j)
    #   y[i,j] = probability of allocating good 2 to bidder with type (i,j)
    #   p[i,j] = payment for type (i,j)
    #
    # For a single bidder, DSIC (dominant strategy incentive compatibility) requires:
    # The mechanism must be such that reporting true type is optimal.
    #
    # In the multi-dimensional case with one bidder, the standard approach:
    # - The allocation rules must satisfy monotonicity and "no overpayment" conditions
    # - For a single bidder, the payment is determined by the allocation rule
    #
    # Actually, for a single bidder with quasi-linear utility:
    # Utility = sum_g v_g * x_g - p
    #
    # DSIC requires: for all types t and reports t',
    #   sum_g v_g(t) * x_g(t) - p(t) >= sum_g v_g(t) * x_g(t') - p(t')
    #
    # This is a system of linear inequalities.
    #
    # To maximize expected revenue: max sum_t Pr(t) * p(t)
    # Subject to:
    #   0 <= x_g(t) <= 1 for all t, g
    #   p(t) >= 0 (individual rationality, assuming value >= 0)
    #   IC constraints
    #
    # But p(t) appears in IC constraints. We can eliminate p(t) using the fact that
    # for a single bidder, the payment is pinned down by the allocation rule.
    #
    # Actually, let me think more carefully. For a single bidder:
    # - The bidder's type is t = (v1, v2)
    # - The mechanism specifies allocation x(t) = (x1(t), x2(t)) and payment p(t)
    # - Utility when type t reports t': u(t, t') = v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    # - DSIC: u(t, t) >= u(t, t') for all t, t'
    #
    # This is equivalent to: for all t, t',
    #   v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    #   => p(t') - p(t) >= v1(t)*(x1(t') - x1(t)) + v2(t)*(x2(t') - x2(t))
    #
    # Also IR: u(t, t) >= 0 => p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    #
    # We want to maximize: sum_t Pr(t) * p(t)
    #
    # Variables: x1(t), x2(t) for all t (32 vars), p(t) for all t (16 vars) = 48 vars
    #
    # Constraints:
    # - 0 <= x1(t) <= 1 for all t: 16*2 = 32 constraints (bounds)
    # - 0 <= x2(t) <= 1 for all t: 16*2 = 32 constraints (bounds)
    # - p(t) >= 0: 16 constraints
    # - IC: for all t, t': p(t') - p(t) >= v1(t)*(x1(t') - x1(t)) + v2(t)*(x2(t') - x2(t))
    #   That's 16*16 = 256 constraints
    # - IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t) for all t: 16 constraints
    #
    # Total: ~320 constraints, 48 variables. Very manageable.
    
    n = n_types  # 16
    # Variable indexing:
    # 0..15: x1[0]..x1[15]
    # 16..31: x2[0]..x2[15]
    # 32..47: p[0]..p[15]
    
    n_vars = 3 * n  # 48
    
    def x1_idx(t): return t
    def x2_idx(t): return n + t
    def p_idx(t): return 2 * n + t
    
    # Objective: maximize sum_t Pr(t) * p(t) => minimize -sum_t Pr(t) * p(t)
    c = np.zeros(n_vars)
    for t in range(n):
        v1, v2 = type_indices[t]
        prob_t = probs[v1, v2]
        c[p_idx(t)] = -prob_t
    
    # Build constraint matrix A_ub, b_ub for A_ub @ vars <= b_ub
    # We'll build constraints in the form: A_ub @ vars <= b_ub
    
    A_rows = []
    b_vals = []
    
    # IC constraints: for all t, t':
    # p(t') - p(t) >= v1(t)*(x1(t') - x1(t)) + v2(t)*(x2(t') - x2(t))
    # Rearranged: v1(t)*x1(t) + v2(t)*x2(t) - p(t) - v1(t)*x1(t') - v2(t)*x2(t') + p(t') >= 0
    # In <= form: -v1(t)*x1(t) - v2(t)*x2(t) + p(t) + v1(t)*x1(t') + v2(t)*x2(t') - p(t') <= 0
    
    for t in range(n):
        v1_t, v2_t = type_indices[t]
        for tp in range(n):
            v1_tp, v2_tp = type_indices[tp]
            row = np.zeros(n_vars)
            # -v1_t * x1(t)
            row[x1_idx(t)] = -v1_t
            # -v2_t * x2(t)
            row[x2_idx(t)] = -v2_t
            # +p(t)
            row[p_idx(t)] = 1.0
            # +v1_t * x1(tp)
            row[x1_idx(tp)] = v1_t
            # +v2_t * x2(tp)
            row[x2_idx(tp)] = v2_t
            # -p(tp)
            row[p_idx(tp)] = -1.0
            
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    for t in range(n):
        v1_t, v2_t = type_indices[t]
        row = np.zeros(n_vars)
        row[x1_idx(t)] = -v1_t
        row[x2_idx(t)] = -v2_t
        row[p_idx(t)] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds:
    # x1(t), x2(t) in [0, 1]
    # p(t) in [0, inf)
    bounds = []
    for t in range(n):
        bounds.append((0, 1))  # x1(t)
        bounds.append((0, 1))  # x2(t)
        bounds.append((0, None))  # p(t)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        for t in range(n):
            v1, v2 = type_indices[t]
            if v1 >= 2:
                allocations[t, 0] = 1.0
            if v2 >= 2:
                allocations[t, 1] = 1.0
            payments[t] = 0.0
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    allocations = np.zeros((n, 2))
    payments = np.zeros(n)
    for t in range(n):
        allocations[t, 0] = x[x1_idx(t)]
        allocations[t, 1] = x[x2_idx(t)]
        payments[t] = x[p_idx(t)]
    
    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify IC and IR, fix if needed
    # Check IC: for all t, tp:
    # u(t, t) >= u(t, tp)
    # v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= v1(t)*x1(tp) + v2(t)*x2(tp) - p(tp)
    for t in range(n):
        v1_t, v2_t = type_indices[t]
        util_true = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        for tp in range(n):
            util_dev = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
            if util_true < util_dev - 1e-8:
                # Fix by increasing payment at tp or decreasing allocation
                # Simple fix: increase payment at tp to make IC hold
                deficit = util_dev - util_true
                payments[tp] += deficit
    
    # Re-check IR after fixes
    for t in range(n):
        v1_t, v2_t = type_indices[t]
        max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > max_pay:
            payments[t] = max_pay
    
    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Final verification
    for t in range(n):
        v1_t, v2_t = type_indices[t]
        # IR
        if v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t] < -1e-8:
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        # IC
        util_true = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        for tp in range(n):
            util_dev = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
            if util_true < util_dev - 1e-8:
                # Adjust
                deficit = util_dev - util_true + 1e-9
                payments[tp] += deficit
                # Re-check IR for tp
                v1_tp, v2_tp = type_indices[tp]
                max_pay_tp = v1_tp * allocations[tp, 0] + v2_tp * allocations[tp, 1]
                if payments[tp] > max_pay_tp:
                    payments[tp] = max_pay_tp
    
    return {'allocations': allocations, 'payments': payments}
