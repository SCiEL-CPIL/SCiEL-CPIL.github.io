# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 70.0  # Safety margin before 90s limit

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

    def validate_and_update(allocations, payments):
        nonlocal best_allocations, best_payments, best_revenue
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)

        # Check IR
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -1e-8:
                return False

        # Check IC
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > 1e-8:
                    return False

        revenue = np.dot(probs, payments)
        if revenue > best_revenue:
            best_revenue = revenue
            best_allocations = allocations.copy()
            best_payments = payments.copy()
        return True

    # --- Approach 1: General IC LP (48 vars) ---
    # Variables: x_i (allocations for good 1), y_i (allocations for good 2), p_i (payments)
    # Objective: max sum p_i * p_i (prob)
    # Constraints:
    # 1. IR: v_i . a_i - p_i >= 0  =>  p_i <= v_i . a_i
    # 2. IC: v_i . a_i - p_i >= v_i . a_j - p_j  =>  p_i - p_j <= v_i . (a_i - a_j)
    # 3. 0 <= a_i, a_j <= 1
    # 4. p_i >= 0

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

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                  'primal_feasibility_tolerance': 1e-10,
                                  'dual_infeasibility_tolerance': 1e-10,
                                  'primal_infeasibility_tolerance': 1e-10,
                                  'ipm_optimality_tolerance': 1e-10})

        if result.success:
            x = result.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = x[2*i]
                allocations[i, 1] = x[2*i+1]
                payments[i] = x[32+i]
            validate_and_update(allocations, payments)

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

            result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                      'primal_feasibility_tolerance': 1e-10,
                                      'dual_infeasibility_tolerance': 1e-10,
                                      'primal_infeasibility_tolerance': 1e-10,
                                      'ipm_optimality_tolerance': 1e-10})

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

                validate_and_update(allocations, payments)

        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 3: Threshold Auction Search ---
    # For single bidder, optimal mechanism is often a threshold auction.
    # Try all combinations of thresholds for each good.
    # A threshold auction allocates good k if v_k >= t_k.
    # Payment is p(v) = sum_k v_k * I(v_k >= t_k) - sum_{k' < t_k} I(v_{k'} >= t_{k'}) ... wait, this is for unit demand.
    # For additive, payment is p(v) = sum_k [v_k * I(v_k >= t_k) - sum_{j < t_k} I(v_j >= t_j)]? No.
    # Payment for additive threshold: p(v) = sum_k (v_k - t_k) * I(v_k >= t_k) + sum_{k' != k} t_{k'} * I(v_{k'} >= t_{k'})?
    # Actually, for additive, the IC constraint implies p(v) = sum_k v_k x_k(v) - u(v).
    # And u(v) is the utility.
    # For a threshold auction with thresholds t_1, t_2:
    # x_1(v) = 1 if v_1 >= t_1 else 0
    # x_2(v) = 1 if v_2 >= t_2 else 0
    # u(v) = u(0,0) + sum_{k=1}^2 (v_k - t_k) I(v_k >= t_k) ? No.
    # u(v_1, v_2) = u(0,0) + integral from 0 to v_1 of x_1(s, v_2) ds + integral from 0 to v_2 of x_2(v_1, s) ds
    # If x_1(s, v_2) = 1 for s >= t_1, then integral is max(0, v_1 - t_1).
    # If x_2(v_1, s) = 1 for s >= t_2, then integral is max(0, v_2 - t_2).
    # So u(v) = u(0,0) + max(0, v_1 - t_1) + max(0, v_2 - t_2).
    # IR requires u(0,0) >= 0. To maximize revenue, set u(0,0) = 0.
    # So p(v) = v_1 x_1(v) + v_2 x_2(v) - u(v)
    # p(v) = v_1 I(v_1 >= t_1) + v_2 I(v_2 >= t_2) - [max(0, v_1 - t_1) + max(0, v_2 - t_2)]
    # If v_1 >= t_1, p_1 part = v_1 - (v_1 - t_1) = t_1.
    # If v_1 < t_1, p_1 part = 0 - 0 = 0.
    # So p(v) = t_1 I(v_1 >= t_1) + t_2 I(v_2 >= t_2).
    # This is a simple threshold auction!
    # Let's search over all possible thresholds t_1, t_2 in {0, 1, 2, 3, 4}.
    # t=4 means never allocate.

    if time.time() < deadline - 5.0:
        try:
            for t1 in range(5):
                for t2 in range(5):
                    allocations = np.zeros((16, 2))
                    payments = np.zeros(16)
                    for i in range(16):
                        v1, v2 = V[i]
                        if v1 >= t1:
                            allocations[i, 0] = 1.0
                        if v2 >= t2:
                            allocations[i, 1] = 1.0
                        payments[i] = t1 * (1.0 if v1 >= t1 else 0.0) + t2 * (1.0 if v2 >= t2 else 0.0)
                    validate_and_update(allocations, payments)
        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 4: High-Precision Polish ---
    # Re-solve the General IC LP with tighter tolerances if we have time.
    if time.time() < deadline - 5.0 and best_revenue > 0:
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

            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                      'primal_feasibility_tolerance': 1e-12,
                                      'dual_infeasibility_tolerance': 1e-12,
                                      'primal_infeasibility_tolerance': 1e-12,
                                      'ipm_optimality_tolerance': 1e-12})

            if result.success:
                x = result.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[2*i]
                    allocations[i, 1] = x[2*i+1]
                    payments[i] = x[32+i]
                validate_and_update(allocations, payments)
        except Exception:
            pass

    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
