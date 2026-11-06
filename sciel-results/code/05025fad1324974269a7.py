# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit

    # 1. Parse Inputs
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    probs = np.clip(probs, 0, 1)

    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]

    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]

    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    best_revenue = 0.0

    def check_validity(allocations, payments, V, tol=1e-8):
        """Check IR and IC constraints"""
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -tol:
                return False
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > tol:
                    return False
        return True

    def extract_solution(x, V):
        """Extract allocations and payments from LP solution vector"""
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            payments[i] = x[32+i]
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        return allocations, payments

    def build_ic_lp(V, probs, tol=1e-10):
        """Build the 48-var IC LP"""
        n_vars = 48
        c = np.zeros(n_vars)
        for i in range(16):
            c[32 + i] = -probs[i]

        A_ir = np.zeros((16, n_vars))
        b_ir = np.zeros(16)
        for i in range(16):
            A_ir[i, 2*i] = -V[i, 0]
            A_ir[i, 2*i+1] = -V[i, 1]
            A_ir[i, 32+i] = 1

        n_ic = 16 * 16
        A_ic = np.zeros((n_ic, n_vars))
        b_ic = np.zeros(n_ic)

        for i in range(16):
            for ip in range(16):
                idx = i * 16 + ip
                A_ic[idx, 2*ip] += V[i, 0]
                A_ic[idx, 2*ip+1] += V[i, 1]
                A_ic[idx, 32+ip] += -1
                A_ic[idx, 2*i] += -V[i, 0]
                A_ic[idx, 2*i+1] += -V[i, 1]
                A_ic[idx, 32+i] += 1

        A = np.vstack([A_ir, A_ic])
        b = np.concatenate([b_ir, b_ic])

        bounds = []
        for i in range(16):
            bounds.append((0, 1))
            bounds.append((0, 1))
        for i in range(16):
            bounds.append((0, None))

        return c, A, b, bounds, tol

    # --- Approach 1: General IC LP with multiple solver configs ---
    for tol in [1e-10, 1e-12, 1e-14]:
        if time.time() > deadline - 5.0:
            break
        try:
            c, A, b, bounds, tol_val = build_ic_lp(V, probs, tol)
            
            options = {
                'presolve': True,
                'dual_feasibility_tolerance': tol_val,
                'primal_feasibility_tolerance': tol_val,
                'dual_infeasibility_tolerance': tol_val,
                'primal_infeasibility_tolerance': tol_val,
                'ipm_optimality_tolerance': tol_val,
                'time_limit': 20.0
            }
            
            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', options=options)
            
            if result.success:
                allocations, payments = extract_solution(result.x, V)
                if check_validity(allocations, payments, V):
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 2: Monotonicity-based LP for grid support (32 vars) ---
    is_grid = True
    for i, (v1, v2) in enumerate(support_list):
        if v1 != i // 4 or v2 != i % 4:
            is_grid = False
            break

    if is_grid:
        try:
            num_vars = 32
            obj_c = np.zeros(num_vars)
            A_ub = []
            b_ub = []

            # Monotonicity Constraints
            for v1 in range(1, 4):
                for v2 in range(4):
                    row = np.zeros(num_vars)
                    idx_curr = v1 * 4 + v2
                    idx_prev = (v1 - 1) * 4 + v2
                    row[idx_curr] = -1
                    row[idx_prev] = 1
                    A_ub.append(row)
                    b_ub.append(0)

            for v1 in range(4):
                for v2 in range(1, 4):
                    row = np.zeros(num_vars)
                    idx_curr = 16 + v1 * 4 + v2
                    idx_prev = 16 + v1 * 4 + (v2 - 1)
                    row[idx_curr] = -1
                    row[idx_prev] = 1
                    A_ub.append(row)
                    b_ub.append(0)

            # Payment Non-Negativity: p(v) >= 0
            for r in range(16):
                v1 = r // 4
                v2 = r % 4
                row = np.zeros(num_vars)
                row[v1 * 4 + v2] = -v1
                row[16 + v1 * 4 + v2] = -v2
                for k in range(1, v1 + 1):
                    prev_v1 = v1 - k
                    row[prev_v1 * 4 + v2] += 1
                for k in range(1, v2 + 1):
                    prev_v2 = v2 - k
                    row[16 + v1 * 4 + prev_v2] += 1
                A_ub.append(row)
                b_ub.append(0)

            # Objective: Maximize Sum P(r) * p(r)
            for r in range(16):
                v1 = r // 4
                v2 = r % 4
                p_r = probs[r]
                obj_c[v1 * 4 + v2] += p_r * v1
                obj_c[16 + v1 * 4 + v2] += p_r * v2
                for k in range(1, v1 + 1):
                    prev_v1 = v1 - k
                    obj_c[prev_v1 * 4 + v2] -= p_r
                for k in range(1, v2 + 1):
                    prev_v2 = v2 - k
                    obj_c[16 + v1 * 4 + prev_v2] -= p_r

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            c_min = -obj_c
            bounds = [(0, 1)] * num_vars

            for tol in [1e-10, 1e-12]:
                if time.time() > deadline - 3.0:
                    break
                options = {
                    'presolve': True,
                    'dual_feasibility_tolerance': tol,
                    'primal_feasibility_tolerance': tol,
                    'dual_infeasibility_tolerance': tol,
                    'primal_infeasibility_tolerance': tol,
                    'ipm_optimality_tolerance': tol,
                    'time_limit': 10.0
                }
                result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)

                if result.success:
                    x_opt = result.x
                    x1_opt = np.clip(x_opt[0:16], 0, 1)
                    x2_opt = np.clip(x_opt[16:32], 0, 1)

                    allocations = np.column_stack((x1_opt, x2_opt))
                    payments = np.zeros(16)

                    for r in range(16):
                        v1 = r // 4
                        v2 = r % 4
                        p_r = v1 * x1_opt[v1 * 4 + v2] + v2 * x2_opt[v1 * 4 + v2]
                        for k in range(1, v1 + 1):
                            prev_v1 = v1 - k
                            p_r -= x1_opt[prev_v1 * 4 + v2]
                        for k in range(1, v2 + 1):
                            prev_v2 = v2 - k
                            p_r -= x2_opt[v1 * 4 + prev_v2]
                        payments[r] = max(0, p_r)

                    if check_validity(allocations, payments, V):
                        revenue = np.dot(probs, payments)
                        if revenue > best_revenue:
                            best_revenue = revenue
                            best_allocations = allocations.copy()
                            best_payments = payments.copy()

        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 3: Utility-based formulation with explicit allocations ---
    # Variables: u_i (16) + a_i1, a_i2 (32) = 48 vars
    # This is equivalent to Approach 1 but with different constraint structure
    if time.time() < deadline - 5.0:
        try:
            n_vars = 48
            # Objective: max sum p_i * (v_i . a_i - u_i)
            # = sum p_i * v_i1 * a_i1 + sum p_i * v_i2 * a_i2 - sum p_i * u_i
            c = np.zeros(n_vars)
            for i in range(16):
                c[2*i] = -probs[i] * V[i, 0]
                c[2*i+1] = -probs[i] * V[i, 1]
                c[32+i] = probs[i]  # min -sum p_i u_i => c[32+i] = p_i for minimization

            A_ub_rows = []
            b_ub_vals = []

            # IR: u_i >= 0 => -u_i <= 0
            for i in range(16):
                row = np.zeros(n_vars)
                row[32+i] = -1
                A_ub_rows.append(row)
                b_ub_vals.append(0)

            # IC: u_i - u_j >= (v_i - v_j) . a_j
            # => (v_i - v_j) . a_j - u_i + u_j <= 0
            for i in range(16):
                for j in range(16):
                    row = np.zeros(n_vars)
                    row[2*j] = V[i, 0] - V[j, 0]
                    row[2*j+1] = V[i, 1] - V[j, 1]
                    row[32+i] = -1
                    row[32+j] = 1
                    A_ub_rows.append(row)
                    b_ub_vals.append(0)

            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_vals)

            bounds = []
            for i in range(16):
                bounds.append((0, 1))
                bounds.append((0, 1))
            for i in range(16):
                bounds.append((0, None))

            options = {
                'presolve': True,
                'dual_feasibility_tolerance': 1e-12,
                'primal_feasibility_tolerance': 1e-12,
                'dual_infeasibility_tolerance': 1e-12,
                'primal_infeasibility_tolerance': 1e-12,
                'ipm_optimality_tolerance': 1e-12,
                'time_limit': 15.0
            }
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)

            if result.success:
                x = result.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[2*i]
                    allocations[i, 1] = x[2*i+1]
                    u_i = x[32+i]
                    payments[i] = np.dot(V[i], allocations[i]) - u_i

                allocations = np.clip(allocations, 0, 1)
                payments = np.maximum(payments, 0)

                if check_validity(allocations, payments, V):
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 4: Post-processing - adjust solution to maximize revenue while maintaining feasibility ---
    if best_revenue > 0 and time.time() < deadline - 3.0:
        try:
            # Try small perturbations to see if we can improve
            # The LP should already be optimal, but numerical issues might leave room
            # Try solving with the objective slightly adjusted to break ties favorably
            
            # Alternative: use the dual solution to identify binding constraints
            # and check if we can shift payments slightly
            
            # For now, just ensure we have the best valid solution
            pass
        except Exception:
            pass

    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
