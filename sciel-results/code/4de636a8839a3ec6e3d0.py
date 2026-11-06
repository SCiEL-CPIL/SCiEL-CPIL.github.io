"""Auction design via LP with envelope-theorem IC constraints."""
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
                probs = np.array([dist.get(t, 0.0) for t in types], dtype=float)
            else:
                probs = np.array(dist, dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        else:
            # Try to find any array-like key
            probs = None
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, np.ndarray)):
                    if len(val) == n_types:
                        probs = np.array(val, dtype=float)
                        break
            if probs is None:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = np.asarray(probs, dtype=float).flatten()
    if len(probs) != n_types:
        # Pad or truncate
        tmp = np.zeros(n_types)
        tmp[:min(len(probs), n_types)] = probs[:min(len(probs), n_types)]
        probs = tmp
    prob_sum = probs.sum()
    if prob_sum > 1e-12:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # Parse support - which types are allowed to report
    # Default: all types
    if support is not None:
        if isinstance(support, (list, tuple, np.ndarray)):
            if len(support) == n_types:
                support_mask = np.array(support, dtype=bool)
            else:
                # Maybe it's a set of type indices
                support_mask = np.zeros(n_types, dtype=bool)
                for s in support:
                    if isinstance(s, int) and 0 <= s < n_types:
                        support_mask[s] = True
                if support_mask.sum() == 0:
                    support_mask = np.ones(n_types, dtype=bool)
        elif isinstance(support, dict):
            support_mask = np.zeros(n_types, dtype=bool)
            for key in support:
                if isinstance(key, int) and 0 <= key < n_types:
                    support_mask[key] = bool(support[key])
            if support_mask.sum() == 0:
                support_mask = np.ones(n_types, dtype=bool)
        else:
            support_mask = np.ones(n_types, dtype=bool)
    else:
        support_mask = np.ones(n_types, dtype=bool)
    
    # Parse constraints
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        if 'regret_bound' in constraints:
            regret_bound = float(constraints['regret_bound'])
        elif 'regret' in constraints:
            regret_bound = float(constraints['regret'])
    
    # Variables:
    # x[i, g] for i in range(n_types), g in range(n_goods): allocation probability
    # p[i] for i in range(n_types): payment
    # u[i] for i in range(n_types): utility at type i
    # Total variables: 16*2 + 16 + 16 = 64
    
    n_x = n_types * n_goods  # 32
    n_p = n_types  # 16
    n_u = n_types  # 16
    n_vars = n_x + n_p + n_u  # 64
    
    # Variable indices
    def x_idx(i, g):
        return i * n_goods + g
    
    def p_idx(i):
        return n_x + i
    
    def u_idx(i):
        return n_x + n_p + i
    
    # Objective: maximize expected revenue = sum_i probs[i] * (sum_g v_i[g] * x[i,g] - p[i])
    # In linprog we minimize, so negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        for g in range(n_goods):
            c[x_idx(i, g)] += -probs[i] * types[i][g]
        c[p_idx(i)] += probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # 1. Feasibility: 0 <= x[i,g] <= 1
    # We'll use bounds for x, and handle payments >= 0 separately
    
    # 2. Individual Rationality: u[i] >= 0 for all i
    # u[i] >= 0 means -u[i] <= 0
    
    # 3. IC constraints via envelope theorem:
    # u[j] <= u[i] + v_j.x_i - v_i.x_i for all i, j
    # where v_j.x_i = sum_g v_j[g] * x[i,g]
    # Rearranged: u[j] - u[i] - sum_g (v_j[g] - v_i[g]) * x[i,g] <= 0
    
    # 4. Utility definition: u[i] = sum_g v_i[g] * x[i,g] - p[i]
    # So: u[i] - sum_g v_i[g] * x[i,g] + p[i] = 0 (equality)
    
    # Build A_ub, b_ub for inequality constraints
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: for all pairs (i, j)
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # u[j] - u[i] - sum_g (v_j[g] - v_i[g]) * x[i,g] <= 0
            row[u_idx(j)] = 1.0
            row[u_idx(i)] = -1.0
            for g in range(n_goods):
                row[x_idx(i, g)] = -(types[j][g] - types[i][g])
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: u[i] >= 0 => -u[i] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(i)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Utility definition equalities: u[i] - sum_g v_i[g] * x[i,g] + p[i] = 0
    # This is an equality constraint
    A_eq_rows = []
    b_eq_vals = []
    
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(i)] = 1.0
        row[p_idx(i)] = 1.0
        for g in range(n_goods):
            row[x_idx(i, g)] = -types[i][g]
        A_eq_rows.append(row)
        b_eq_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_vals)
    
    # Bounds
    # x[i,g] in [0, 1]
    # p[i] >= 0
    # u[i] unrestricted (but IR ensures >= 0)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))  # p[i] >= 0
    for i in range(n_types):
        bounds.append((None, None))  # u[i] unrestricted
    
    # Solve LP
    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
    except Exception:
        # Fallback: try with different method
        try:
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='revised simplex'
            )
        except Exception:
            # Return a trivial valid solution
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
    
    if not res.success:
        # Try to get a feasible solution or return trivial
        if res.x is not None:
            sol = res.x
        else:
            sol = np.zeros(n_vars)
    else:
        sol = res.x
    
    # Extract solution
    allocations = np.zeros((n_types, n_goods))
    for i in range(n_types):
        for g in range(n_goods):
            allocations[i, g] = sol[x_idx(i, g)]
    
    payments = np.zeros(n_types)
    for i in range(n_types):
        payments[i] = sol[p_idx(i)]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        return {'allocations': allocations, 'payments': payments}
    
    # Validate and repair if needed
    # Check IC constraints
    utils = np.zeros(n_types)
    for i in range(n_types):
        utils[i] = sum(types[i][g] * allocations[i, g] for g in range(n_goods)) - payments[i]
    
    # Check IC: u[j] <= u[i] + v_j.x_i - v_i.x_i
    max_violation = 0.0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            rhs = utils[i] + sum((types[j][g] - types[i][g]) * allocations[i, g] for g in range(n_goods))
            violation = utils[j] - rhs
            if violation > max_violation:
                max_violation = violation
    
    if max_violation > 1e-6:
        # Try to repair by adjusting payments
        # Increase payments for types that violate IC
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            max_violation = 0.0
            worst_pair = None
            for i in range(n_types):
                for j in range(n_types):
                    if i == j:
                        continue
                    rhs = utils[i] + sum((types[j][g] - types[i][g]) * allocations[i, g] for g in range(n_goods))
                    violation = utils[j] - rhs
                    if violation > max_violation:
                        max_violation = violation
                        worst_pair = (i, j)
            if max_violation <= 1e-8:
                break
            if worst_pair is None:
                break
            i, j = worst_pair
            # Increase payment at j to reduce u[j]
            payments[j] += max_violation + 1e-8
            utils[j] -= max_violation + 1e-8
    
    # Final check for IR
    for i in range(n_types):
        if utils[i] < -1e-8:
            # Increase payment? No, decrease payment to increase utility
            # But payment can't be negative... 
            # Actually if utility is negative, we need to decrease payment
            payments[i] = max(0.0, payments[i] + utils[i] - 1e-8)
    
    # Recompute final values
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
