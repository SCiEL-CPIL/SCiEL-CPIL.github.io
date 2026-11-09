import numpy as np
from scipy.optimize import linprog
import time
import sys

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        # Ensure no negative probs
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # LP Formulation with epigraph variable t
    # Variables: x1[0..15], x2[16..31], p[32..47], t[48]
    # t represents the minimum utility across all types (epigraph)
    # IC: u(t) >= t for all t => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) - t <= 0
    # This subsumes IR (since t >= 0) and IC (u(t) >= u(t') for all t,t')
    # Actually, let's use the standard form:
    # u(t) = v1(t)*x1(t) + v2(t)*x2(t) - p(t)
    # IC: u(t) >= u(t') for all t, t'
    # IR: u(t) >= 0 for all t
    # 
    # Epigraph: introduce t such that t <= u(t) for all t, maximize revenue - 0*t
    # Since we maximize revenue and t is free (unbounded below), t will go to -inf
    # unless constrained. Better approach: just use the direct constraints.
    # 
    # Let's stick with the direct formulation but use tighter tolerances
    # and add a polish pass.
    
    n_vars = 48
    
    A_rows = []
    b_rows = []
    
    # IC: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    # => -v1(t)*x1(t) + v1(t)*x1(t') - v2(t)*x2(t) + v2(t)*x2(t') + p(t) - p(t') <= 0
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve with tight tolerances
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_objective_tolerance': 1e-12,
            'primal_objective_tolerance': 1e-12,
            'time_limit': 50
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # Polish pass: re-solve with the current solution as warm start context
    # Use tighter tolerances and verify optimality
    elapsed = time.time() - start_time
    if elapsed < 60.0 and result is not None and result.success:
        try:
            options2 = {
                'presolve': True,
                'dual_feasibility_tolerance': 1e-14,
                'primal_feasibility_tolerance': 1e-14,
                'dual_objective_tolerance': 1e-14,
                'primal_objective_tolerance': 1e-14,
                'time_limit': 25
            }
            result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                            method='highs', options=options2)
            if result2 is not None and result2.success:
                x2_sol = result2.x
                x1_p = np.clip(x2_sol[0:16], 0, 1)
                x2_p = np.clip(x2_sol[16:32], 0, 1)
                p_p = np.maximum(x2_sol[32:48], 0)
                allocs_p = np.column_stack((x1_p, x2_p))
                pays_p = p_p
                # Compare revenue
                rev_orig = np.sum(probs * payments)
                rev_polish = np.sum(probs * pays_p)
                if rev_polish >= rev_orig - 1e-10:
                    allocations = allocs_p
                    payments = pays_p
        except Exception:
            pass
    
    # Verification
    def check_violations(allocs, pays):
        """Returns max IC violation and max IR violation."""
        max_ic_viol = 0.0
        max_ir_viol = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            util_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            if util_t < 0:
                max_ir_viol = max(max_ir_viol, -util_t)
            u_t = util_t
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_ic_viol:
                        max_ic_viol = viol
        return max_ic_viol, max_ir_viol

    max_ic_viol, max_ir_viol = check_violations(allocations, payments)

    # Only apply minimal fixes if LP solution has numerical issues
    if max_ic_viol > 1e-9 or max_ir_viol > 1e-9:
        # Fix IR violations
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            if payments[t] > max_pay + 1e-12:
                payments[t] = max(0.0, max_pay)

        # Fix IC violations by adjusting payments minimally
        for _ in range(30):
            max_ic_viol, max_ir_viol = check_violations(allocations, payments)
            if max_ic_viol < 1e-10 and max_ir_viol < 1e-10:
                break
            if max_ic_viol > 1e-10:
                worst_viol = 0.0
                worst_t = -1
                worst_t2 = -1
                for t in range(16):
                    v1_t, v2_t = type_to_vals[t]
                    u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                    for t2 in range(16):
                        if t == t2:
                            continue
                        u_t2_as_t = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
                        if u_t < u_t2_as_t:
                            viol = u_t2_as_t - u_t
                            if viol > worst_viol:
                                worst_viol = viol
                                worst_t = t
                                worst_t2 = t2
                if worst_t >= 0:
                    v1_w, v2_w = type_to_vals[worst_t]
                    required = v1_w * allocations[worst_t, 0] + v2_w * allocations[worst_t, 1] \
                             - v1_w * allocations[worst_t2, 0] - v2_w * allocations[worst_t2, 1] + payments[worst_t2]
                    max_pay_w = v1_w * allocations[worst_t, 0] + v2_w * allocations[worst_t, 1]
                    if required > payments[worst_t] and required <= max_pay_w + 1e-12:
                        payments[worst_t] = required
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                if payments[t] > max_pay + 1e-12:
                    payments[t] = max(0.0, max_pay)

        max_ic_viol, max_ir_viol = check_violations(allocations, payments)
        if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)

    return {'allocations': allocations, 'payments': payments}
