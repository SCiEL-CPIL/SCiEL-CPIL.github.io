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
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.array([dist.get(str(t), 0.0) for t in types], dtype=float)
            else:
                probs = np.array(dist, dtype=float)
        else:
            # Default: uniform
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Support: list of allowed reports (types)
    if support is None:
        support = list(range(n_types))
    elif isinstance(support, (list, tuple)):
        # If support contains tuples, map to indices
        if len(support) > 0 and isinstance(support[0], tuple):
            support_idx = []
            for s in support:
                for i, t in enumerate(types):
                    if t == s:
                        support_idx.append(i)
                        break
            support = support_idx
        else:
            support = list(support)
    else:
        support = list(range(n_types))
    
    support_set = set(support)
    n_support = len(support)
    if n_support == 0:
        n_support = n_types
        support_set = set(range(n_types))
    
    # Parse constraints
    max_regret = 1e-8
    if isinstance(constraints, dict):
        max_regret = constraints.get('max_regret', constraints.get('regret', max_regret))
    
    # Build the LP for mechanism design
    # Variables: x[t][g] for allocation (probability of allocating good g to type t)
    #             p[t] for payment to type t
    # Total variables: n_types * n_goods + n_types = 16*2 + 16 = 48
    
    # Objective: maximize expected revenue = sum_t p[t] * probs[t]
    # Subject to:
    # 1. Feasibility: 0 <= x[t][g] <= 1
    # 2. Individual rationality: u[t] = sum_g v[t][g]*x[t][g] - p[t] >= 0
    # 3. Incentive compatibility: u[t] >= sum_g v[t'][g]*x[t][g] - p[t] for all t, t' in support
    #    i.e., u[t] - u[t'] >= sum_g (v[t'][g] - v[t][g]) * x[t][g]
    #    Or equivalently: sum_g v[t][g]*x[t][g] - p[t] >= sum_g v[t'][g]*x[t][g] - p[t']
    
    # Let's use the utility-based formulation for cleaner IC constraints
    # Variables: u[t] (utility), x[t][g] (allocation prob), p[t] (payment)
    # Actually, let's keep it simple with direct allocation and payment variables.
    
    # Variable indexing:
    # x[t][g] is at index t*n_goods + g
    # p[t] is at index n_types*n_goods + t
    n_vars = n_types * n_goods + n_types
    
    def x_idx(t, g):
        return t * n_goods + g
    
    def p_idx(t):
        return n_types * n_goods + t
    
    # Objective: maximize sum_t p[t] * probs[t]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]
    
    # Constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # Feasibility: x[t][g] <= 1 (lower bound 0 handled by bounds)
    # IC constraints: for each t (truthful), t' (deviation) in support:
    # u[t] >= u[t'] where u[t] = sum_g v[t][g]*x[t][g] - p[t]
    # sum_g v[t][g]*x[t][g] - p[t] >= sum_g v[t'][g]*x[t][g] - p[t']
    # sum_g v[t][g]*x[t][g] - sum_g v[t'][g]*x[t][g] - p[t] + p[t'] >= 0
    # -(sum_g v[t][g]*x[t][g] - sum_g v[t'][g]*x[t][g] - p[t] + p[t']) <= 0
    
    for t in support_set:
        for t_prime in support_set:
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            for g in range(n_goods):
                row[x_idx(t, g)] += values[types[t][g]] if False else types[t][g]
                row[x_idx(t_prime, g)] -= types[t_prime][g]
            row[p_idx(t)] -= 1
            row[p_idx(t_prime)] += 1
            # We want: row @ vars >= 0, i.e., -row @ vars <= 0
            A_ub_rows.append(-row)
            b_ub_rows.append(0.0)
    
    # IR constraints: u[t] >= 0 for t in support
    # sum_g v[t][g]*x[t][g] - p[t] >= 0
    # -(sum_g v[t][g]*x[t][g] - p[t]) <= 0
    for t in support_set:
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[x_idx(t, g)] = -types[t][g]
        row[p_idx(t)] = 1
        # We want: sum_g v[t][g]*x[t][g] - p[t] >= 0
        # i.e., -(sum_g v[t][g]*x[t][g] - p[t]) <= 0
        # row as defined: -sum_g v[t][g]*x[t][g] + p[t] <= 0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)
    
    # Bounds: 0 <= x[t][g] <= 1, 0 <= p[t] (payments nonnegative)
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple first-price-like auction
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for t in range(n_types):
            v = types[t]
            if v[0] >= v[1]:
                allocations[t, 0] = 1
                payments[t] = v[1]
            else:
                allocations[t, 1] = 1
                payments[t] = v[0]
        return {'allocations': allocations, 'payments': payments}
    
    x_sol = result.x
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for t in range(n_types):
        for g in range(n_goods):
            allocations[t, g] = x_sol[x_idx(t, g)]
        payments[t] = x_sol[p_idx(t)]
    
    # Verify constraints
    max_regret_found = 0.0
    for t in support_set:
        u_t = sum(types[t][g] * allocations[t, g] for g in range(n_goods)) - payments[t]
        for t_prime in support_set:
            u_dev = sum(types[t_prime][g] * allocations[t, g] for g in range(n_goods)) - payments[t]
            regret = u_dev - u_t
            if regret > max_regret_found:
                max_regret_found = regret
    
    if max_regret_found > max_regret:
        # Fallback
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for t in range(n_types):
            v = types[t]
            if v[0] >= v[1]:
                allocations[t, 0] = 1
                payments[t] = v[1]
            else:
                allocations[t, 1] = 1
                payments[t] = v[0]
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
