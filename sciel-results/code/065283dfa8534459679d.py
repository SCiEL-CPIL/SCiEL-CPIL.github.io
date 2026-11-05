"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Values for each good: {0, 1, 2, 3}
    vals = [0, 1, 2, 3]
    # All 16 types: (v1, v2)
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    n_goods = 2

    # Map type to index
    type_idx = {t: i for i, t in enumerate(types)}

    # Parse distribution_spec to get probabilities
    # distribution_spec could be a dict mapping types to probs, or a list
    if isinstance(distribution_spec, dict):
        # Keys might be tuples or strings
        probs = np.zeros(n_types)
        for key, p in distribution_spec.items():
            if isinstance(key, (tuple, list)):
                t = tuple(key)
                if t in type_idx:
                    probs[type_idx[t]] = p
            elif isinstance(key, str):
                # Try to parse like "v1,v2" or "v1 v2"
                try:
                    parts = key.replace('(', '').replace(')', '').split(',')
                    parts = [p.strip() for p in parts]
                    t = (int(parts[0]), int(parts[1]))
                    if t in type_idx:
                        probs[type_idx[t]] = p
                except:
                    pass
        # Normalize if needed
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.asarray(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            probs = arr / arr.sum() if arr.sum() > 0 else np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    # Ensure support is respected - if support is given, only those types have nonzero prob
    if support is not None and len(support) > 0:
        # support might be list of tuples
        support_set = set()
        for s in support:
            if isinstance(s, (tuple, list)):
                support_set.add(tuple(s))
            elif isinstance(s, str):
                try:
                    parts = s.replace('(', '').replace(')', '').split(',')
                    parts = [p.strip() for p in parts]
                    support_set.add((int(parts[0]), int(parts[1])))
                except:
                    pass
        if support_set:
            mask = np.zeros(n_types, dtype=bool)
            for t in support_set:
                if t in type_idx:
                    mask[type_idx[t]] = True
            probs[~mask] = 0.0
            s = probs.sum()
            if s > 0:
                probs = probs / s
            else:
                probs = np.ones(n_types) / n_types

    # LP formulation
    # Variables: x = [q_00_0, q_00_1, q_01_0, q_01_1, ..., q_33_0, q_33_1, p_00, p_01, ..., p_33]
    # First 32 vars: allocations q_{type,good}
    # Next 16 vars: payments p_type
    n_alloc = n_types * n_goods  # 32
    n_pay = n_types  # 16
    n_vars = n_alloc + n_pay  # 48

    # Objective: maximize sum_v probs[v] * p_v
    # In linprog, minimize -objective
    c = np.zeros(n_vars)
    for v in range(n_types):
        c[n_alloc + v] = -probs[v]

    # Inequality constraints: A_ub @ x <= b_ub
    constraints_list = []
    b_list = []

    # IC constraints: for each true type v and report v' (v != v'):
    # p_v - p_{v'} <= v . (q_v - q_{v'})
    # p_v - p_{v'} - v1*(q_v1 - q_v1') - v2*(q_v2 - q_v2') <= 0
    for v_true in range(n_types):
        vt = types[v_true]
        for v_rep in range(n_types):
            if v_rep == v_true:
                continue
            vr = types[v_rep]
            row = np.zeros(n_vars)
            # p_v_true coefficient: +1
            row[n_alloc + v_true] = 1.0
            # p_v_rep coefficient: -1
            row[n_alloc + v_rep] = -1.0
            # q_v_true_good1: -vt[0]
            row[v_true * n_goods + 0] = -vt[0]
            # q_v_rep_good1: +vt[0]
            row[v_rep * n_goods + 0] = vt[0]
            # q_v_true_good2: -vt[1]
            row[v_true * n_goods + 1] = -vt[1]
            # q_v_rep_good2: +vt[1]
            row[v_rep * n_goods + 1] = vt[1]
            constraints_list.append(row)
            b_list.append(0.0)

    # IR constraints: p_v <= v . q_v  =>  p_v - v1*q_v1 - v2*q_v2 <= 0
    for v in range(n_types):
        vt = types[v]
        row = np.zeros(n_vars)
        row[n_alloc + v] = 1.0
        row[v * n_goods + 0] = -vt[0]
        row[v * n_goods + 1] = -vt[1]
        constraints_list.append(row)
        b_list.append(0.0)

    # Nonnegativity: p_v >= 0 => -p_v <= 0
    for v in range(n_types):
        row = np.zeros(n_vars)
        row[n_alloc + v] = -1.0
        constraints_list.append(row)
        b_list.append(0.0)

    # Check time before solving
    if time.time() > deadline - 5.0:
        return {
            'allocations': np.zeros((n_types, n_goods)),
            'payments': np.zeros(n_types)
        }

    # Allocation bounds: 0 <= q_{v,g} <= 1
    # These are handled by bounds in linprog
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))

    if constraints_list:
        A_ub = np.array(constraints_list)
        b_ub = np.array(b_list)
    else:
        A_ub = None
        b_ub = None

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            allocations = x[:n_alloc].reshape(n_types, n_goods)
            payments = x[n_alloc:]
            # Clip to ensure feasibility
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            # Verify IC constraints (regret <= 1e-8)
            max_regret = 0.0
            for v_true in range(n_types):
                vt = types[v_true]
                for v_rep in range(n_types):
                    vr = types[v_rep]
                    # Utility for true type reporting v_rep:
                    # u = vt[0]*q[v_rep,0] + vt[1]*q[v_rep,1] - p[v_rep]
                    # Utility for true type reporting truthfully:
                    # u_truth = vt[0]*q[v_true,0] + vt[1]*q[v_true,1] - p[v_true]
                    # Regret = u - u_truth (should be <= 0)
                    u_rep = vt[0] * allocations[v_rep, 0] + vt[1] * allocations[v_rep, 1] - payments[v_rep]
                    u_truth = vt[0] * allocations[v_true, 0] + vt[1] * allocations[v_true, 1] - payments[v_true]
                    regret = u_rep - u_truth
                    if regret > max_regret:
                        max_regret = regret
            
            # If regret is slightly above tolerance, adjust payments
            if max_regret > 1e-8:
                # Add a small uniform payment adjustment to fix IC
                # This may reduce revenue but ensures feasibility
                adjustment = max_regret + 1e-9
                payments = payments + adjustment
                # Re-verify
                max_regret = 0.0
                for v_true in range(n_types):
                    vt = types[v_true]
                    for v_rep in range(n_types):
                        vr = types[v_rep]
                        u_rep = vt[0] * allocations[v_rep, 0] + vt[1] * allocations[v_rep, 1] - payments[v_rep]
                        u_truth = vt[0] * allocations[v_true, 0] + vt[1] * allocations[v_true, 1] - payments[v_true]
                        regret = u_rep - u_truth
                        if regret > max_regret:
                            max_regret = regret
                
                if max_regret > 1e-8:
                    # Fallback to trivial mechanism if still infeasible
                    allocations = np.zeros((n_types, n_goods))
                    payments = np.zeros(n_types)
            
            # Check IR constraints
            for v in range(n_types):
                vt = types[v]
                ir_violation = payments[v] - (vt[0] * allocations[v, 0] + vt[1] * allocations[v, 1])
                if ir_violation > 1e-8:
                    # Fix IR violation by reducing payment
                    payments[v] = max(0, vt[0] * allocations[v, 0] + vt[1] * allocations[v, 1])
            
            # Final check: ensure nonnegative payments
            payments = np.clip(payments, 0, None)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # Fallback: trivial mechanism
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            return {
                'allocations': allocations,
                'payments': payments
            }
    except Exception:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {
            'allocations': allocations,
            'payments': payments
        }
