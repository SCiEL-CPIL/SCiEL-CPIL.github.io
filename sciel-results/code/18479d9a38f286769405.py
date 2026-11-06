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

    # --- Approach 1: General IC LP (48 vars) ---
    try:
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

        # Use tight tolerances for better precision
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})

        if result.success:
            x = result.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = x[2*i]
                allocations[i, 1] = x[2*i+1]
                payments[i] = x[32+i]

            # Polish pass: try to increase payments slightly while maintaining constraints
            # We can do a small local search on payments
            # But first verify validity
            valid = True
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    valid = False
                    break

            if valid:
                for i in range(16):
                    for ip in range(16):
                        util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                        util_truth = np.dot(V[i], allocations[i]) - payments[i]
                        if util_report - util_truth > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                revenue = np.dot(probs, payments)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_allocations = allocations.copy()
                    best_payments = payments.copy()

    except Exception:
        pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 2: Monotonicity LP (32 vars) ---
    # This approach is faster and can be more stable. It assumes the support is a grid.
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
            # x1(v1, v2) >= x1(v1-1, v2) => -x1(v1, v2) + x1(v1-1, v2) <= 0
            for v1 in range(1, 4):
                for v2 in range(4):
                    row = np.zeros(num_vars)
                    idx_curr = v1 * 4 + v2
                    idx_prev = (v1 - 1) * 4 + v2
                    row[idx_curr] = -1
                    row[idx_prev] = 1
                    A_ub.append(row)
                    b_ub.append(0)

            # x2(v1, v2) >= x2(v1, v2-1) => -x2(v1, v2) + x2(v1, v2-1) <= 0
            for v1 in range(4):
                for v2 in range(1, 4):
                    row = np.zeros(num_vars)
                    idx_curr = 16 + v1 * 4 + v2
                    idx_prev = 16 + v1 * 4 + (v2 - 1)
                    row[idx_curr] = -1
                    row[idx_prev] = 1
                    A_ub.append(row)
                    b_ub.append(0)

            # Payment Non-Negativity: p(v) >= 0 => -p(v) <= 0
            # p(v1, v2) = v1*x1(v1, v2) + v2*x2(v1, v2) - sum_{k=1}^{v1} x1(k-1, v2) - sum_{k=1}^{v2} x2(v1, k-1)
            for r in range(16):
                v1 = r // 4
                v2 = r % 4
                row = np.zeros(num_vars)

                # -v1*x1(v1, v2) - v2*x2(v1, v2)
                row[v1 * 4 + v2] = -v1
                row[16 + v1 * 4 + v2] = -v2

                # + sum x1(prev)
                for k in range(1, v1 + 1):
                    prev_v1 = v1 - k
                    row[prev_v1 * 4 + v2] += 1

                # + sum x2(prev)
                for k in range(1, v2 + 1):
                    prev_v2 = v2 - k
                    row[16 + v1 * 4 + prev_v2] += 1

                A_ub.append(row)
                b_ub.append(0)

            # Objective: Maximize Sum P(r) * p(r)
            # p(r) = v1*x1(r) + v2*x2(r) - S1 - S2
            for r in range(16):
                v1 = r // 4
                v2 = r % 4
                p_r = probs[r]

                # Coeff for x1(r): v1 * p_r
                obj_c[v1 * 4 + v2] += p_r * v1
                # Coeff for x2(r): v2 * p_r
                obj_c[16 + v1 * 4 + v2] += p_r * v2

                # Coeff for x1(prev): -p_r
                for k in range(1, v1 + 1):
                    prev_v1 = v1 - k
                    obj_c[prev_v1 * 4 + v2] -= p_r

                # Coeff for x2(prev): -p_r
                for k in range(1, v2 + 1):
                    prev_v2 = v2 - k
                    obj_c[16 + v1 * 4 + prev_v2] -= p_r

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            c_min = -obj_c
            bounds = [(0, 1)] * num_vars

            result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                      'primal_feasibility_tolerance': 1e-12})

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

                # Verify
                valid = True
                for i in range(16):
                    util = np.dot(V[i], allocations[i]) - payments[i]
                    if util < -1e-8:
                        valid = False
                        break
                if valid:
                    for i in range(16):
                        for ip in range(16):
                            util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                            util_truth = np.dot(V[i], allocations[i]) - payments[i]
                            if util_report - util_truth > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break

                if valid:
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()

        except Exception:
            pass

    # --- Approach 3: Utility-based LP with envelope constraints ---
    # Variables: x_{i,k} for i=0..15, k=0,1 (32 vars) and u_i for i=0..15 (16 vars)
    # Total: 48 vars
    # Objective: Maximize sum_i probs[i] * (V_i . x_i - u_i)
    # Constraints:
    # 1. x_{i,k} in [0,1]
    # 2. u_i >= 0 (IR)
    # 3. u_i >= u_j + (V_i - V_j) . x_j for all i,j (IC)
    #    This is equivalent to: u_i - u_j - (V_i - V_j) . x_j >= 0
    #    Or: -u_i + u_j + (V_i - V_j) . x_j <= 0
    
    if time.time() < deadline:
        try:
            n_vars = 48  # 32 allocation + 16 utility
            c = np.zeros(n_vars)
            
            # Objective: Maximize sum_i probs[i] * (V_i . x_i - u_i)
            # = sum_i probs[i] * (sum_k V[i,k]*x[i,k] - u_i)
            # Minimize negative: -sum_i probs[i] * (sum_k V[i,k]*x[i,k] - u_i)
            for i in range(16):
                for k in range(2):
                    c[2*i + k] = -probs[i] * V[i, k]
                c[32 + i] = probs[i]  # -(-u_i * probs[i]) = +u_i * probs[i] in minimization

            A_ub = []
            b_ub = []
            
            # IC constraints: u_i - u_j - (V_i - V_j) . x_j >= 0
            # => -u_i + u_j + (V_i - V_j) . x_j <= 0
            # Note: (V_i - V_j) . x_j = sum_k (V[i,k] - V[j,k]) * x[j,k]
            # So constraint is: -u_i + u_j + sum_k (V[i,k] - V[j,k]) * x[j,k] <= 0
            # Variables: x[j,k] is at index 2*j+k, u_j is at 32+j, u_i is at 32+i
            
            for i in range(16):
                for j in range(16):
                    if i == j:
                        continue
                    row = np.zeros(n_vars)
                    row[32 + i] = -1
                    row[32 + j] = 1
                    for k in range(2):
                        row[2*j + k] = V[i, k] - V[j, k]
                    A_ub.append(row)
                    b_ub.append(0)
            
            # IR constraints: u_i >= 0 => -u_i <= 0
            for i in range(16):
                row = np.zeros(n_vars)
                row[32 + i] = -1
                A_ub.append(row)
                b_ub.append(0)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            bounds = []
            for i in range(32):
                bounds.append((0, 1))
            for i in range(16):
                bounds.append((0, None))
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                      'primal_feasibility_tolerance': 1e-12})
            
            if result.success:
                x = result.x
                allocations = np.zeros((16, 2))
                utilities = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[2*i]
                    allocations[i, 1] = x[2*i+1]
                    utilities[i] = x[32+i]
                
                payments = np.zeros(16)
                for i in range(16):
                    payments[i] = np.dot(V[i], allocations[i]) - utilities[i]
                
                # Verify
                valid = True
                for i in range(16):
                    util = np.dot(V[i], allocations[i]) - payments[i]
                    if util < -1e-8:
                        valid = False
                        break
                if valid:
                    for i in range(16):
                        for ip in range(16):
                            util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                            util_truth = np.dot(V[i], allocations[i]) - payments[i]
                            if util_report - util_truth > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()
                        
        except Exception:
            pass

    # --- Polish Pass: Try to increase payments slightly ---
    if best_revenue > 0 and time.time() < deadline:
        try:
            # Try small perturbations to payments
            # We can try to increase each payment by a small epsilon
            # and check if constraints still hold
            current_allocs = best_allocations.copy()
            current_pays = best_payments.copy()
            
            # Try increasing payments by small amounts
            for attempt in range(50):
                improved = False
                for i in range(16):
                    if probs[i] > 1e-10:
                        # Try small increase
                        delta = 1e-6 / (attempt + 1)
                        new_pays = current_pays.copy()
                        new_pays[i] += delta
                        
                        # Check IR
                        valid = True
                        for j in range(16):
                            util = np.dot(V[j], current_allocs[j]) - new_pays[j]
                            if util < -1e-8:
                                valid = False
                                break
                        
                        if valid:
                            # Check IC
                            for j in range(16):
                                for k in range(16):
                                    util_report = np.dot(V[j], current_allocs[k]) - new_pays[k]
                                    util_truth = np.dot(V[j], current_allocs[j]) - new_pays[j]
                                    if util_report - util_truth > 1e-8:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        
                        if valid:
                            current_pays = new_pays
                            improved = True
                
                if not improved:
                    break
            
            new_revenue = np.dot(probs, current_pays)
            if new_revenue > best_revenue:
                best_revenue = new_revenue
                best_payments = current_pays
        except Exception:
            pass

    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
