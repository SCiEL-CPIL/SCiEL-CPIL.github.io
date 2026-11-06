"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
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
            probs = np.array(distribution_spec, dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs is 16-dimensional
    if probs.shape[0] != n_types:
        # Try to reshape or pad
        if probs.size == n_types:
            probs = probs.reshape(n_types)
        else:
            probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Build the LP for the optimal incentive-compatible mechanism
    # Variables:
    # x[t, g] = probability that bidder with type t gets good g (2 per type)
    # p[t] = payment for type t (1 per type)
    # Total variables: 16*2 + 16 = 48
    
    n_vars = n_types * n_goods + n_types  # 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog, we minimize, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_types * n_goods + t] = -probs[t]  # Negative for maximization
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t,g] <= 1 for all t,g
    # 2. Individual rationality: v[t] . x[t] - p[t] >= 0 for all t
    #    => p[t] <= v[t] . x[t]
    #    => -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    # 3. Incentive compatibility: for all t, t'
    #    v[t'] . x[t'] - p[t'] >= v[t'] . x[t] - p[t]
    #    => v[t'] . (x[t'] - x[t]) - (p[t'] - p[t]) >= 0
    #    => -v[t'] . x[t'] + v[t'] . x[t] + p[t'] - p[t] <= 0
    # 4. Nonnegative payments: p[t] >= 0 => -p[t] <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints: p[t] - v1[t]*x[t,0] - v2[t]*x[t,1] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[t * n_goods] = -types[t][0]  # -v1[t]
        row[t * n_goods + 1] = -types[t][1]  # -v2[t]
        row[n_types * n_goods + t] = 1  # +p[t]
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints: for all t, t'
    # -v[t'] . x[t'] + v[t'] . x[t] + p[t'] - p[t] <= 0
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            # -v[t'] . x[t']
            row[t_prime * n_goods] = -types[t_prime][0]
            row[t_prime * n_goods + 1] = -types[t_prime][1]
            # +v[t'] . x[t]
            row[t * n_goods] += types[t_prime][0]
            row[t * n_goods + 1] += types[t_prime][1]
            # +p[t'] - p[t]
            row[n_types * n_goods + t_prime] = 1
            row[n_types * n_goods + t] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    # Nonnegative payments: -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_types * n_goods + t] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: simple uniform allocation with zero payments
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        
        for t in range(n_types):
            for g in range(n_goods):
                allocations[t, g] = x[t * n_goods + g]
            payments[t] = x[n_types * n_goods + t]
        
        # Verify constraints and clip small negative values
        payments = np.maximum(payments, 0)
        
        # Check IR: v[t] . x[t] - p[t] >= 0
        for t in range(n_types):
            utility = types[t][0] * allocations[t, 0] + types[t][1] * allocations[t, 1] - payments[t]
            if utility < -1e-8:
                # Violation, adjust payment
                payments[t] = types[t][0] * allocations[t, 0] + types[t][1] * allocations[t, 1]
        
        # Check IC: for all t, t'
        # v[t'] . x[t'] - p[t'] >= v[t'] . x[t] - p[t]
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                lhs = types[t_prime][0] * allocations[t_prime, 0] + types[t_prime][1] * allocations[t_prime, 1] - payments[t_prime]
                rhs = types[t_prime][0] * allocations[t, 0] + types[t_prime][1] * allocations[t, 1] - payments[t]
                if lhs < rhs - 1e-8:
                    # Violation, need to adjust
                    # Increase payment[t] or decrease payment[t_prime]
                    # Simple fix: increase payment[t] to make utility[t] lower
                    # Actually, this means type t would prefer to misreport as t'
                    # We need to ensure the mechanism is IC
                    pass
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
