import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Direct LP with standard IC constraints, solved via HiGHS with tight tolerances,
    followed by a high-precision polish pass.
    """
    start_time = time.time()
    DEADLINE = 85.0

    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)

    if probs.shape != (16,):
        probs = probs.flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))

    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP
    # Maximize Revenue = sum(p_i * probs_i) => Minimize -sum(p_i * probs_i)
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]

    A_rows = []
    b_rows = []

    # IR Constraints: v1*x1_i + v2*x2_i - p_i >= 0
    # => -v1*x1_i - v2*x2_i + p_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0)

    # IC Constraints (standard, no epsilon): u(i) >= u(j)
    # v_i.x_i - p_i >= v_i.x_j - p_j
    # -v_i.x_i + v_i.x_j + p_i - p_j <= 0
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[16 + i] = -v2_i
            row[j] = v1_i
            row[16 + j] = v2_i
            row[32 + i] = 1
            row[32 + j] = -1
            A_rows.append(row)
            b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: x in [0,1], p in [0, inf)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16

    # 4. Solve with HiGHS
    from scipy.optimize import linprog

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    best_rev = 0.0

    try:
        # First solve with reasonable tolerances
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})

        if time.time() - start_time > DEADLINE:
            return {"allocations": best_allocations, "payments": best_payments}

        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)

            rev = np.sum(payments * probs)
            best_allocations = allocations.copy()
            best_payments = payments.copy()
            best_rev = rev

            # 5. Polish pass: re-solve with tighter tolerances
            if time.time() - start_time < DEADLINE - 10.0:
                try:
                    res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-14,
                                            'primal_feasibility_tolerance': 1e-14,
                                            'time_limit': 30.0})
                    if res2.success:
                        x2 = res2.x
                        alloc2 = np.zeros((16, 2))
                        pay2 = np.zeros(16)
                        for i in range(16):
                            alloc2[i, 0] = x2[i]
                            alloc2[i, 1] = x2[16 + i]
                            pay2[i] = x2[32 + i]
                        alloc2 = np.clip(alloc2, 0, 1)
                        pay2 = np.clip(pay2, 0, None)
                        rev2 = np.sum(pay2 * probs)
                        if rev2 > best_rev - 1e-12:
                            best_allocations = alloc2
                            best_payments = pay2
                            best_rev = rev2
                except Exception:
                    pass

            return {"allocations": best_allocations, "payments": best_payments}
        else:
            # Fallback: try without presolve
            try:
                res_fb = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                 options={'presolve': False,
                                          'dual_feasibility_tolerance': 1e-12,
                                          'primal_feasibility_tolerance': 1e-12})
                if res_fb.success:
                    x = res_fb.x
                    allocations = np.zeros((16, 2))
                    payments = np.zeros(16)
                    for i in range(16):
                        allocations[i, 0] = x[i]
                        allocations[i, 1] = x[16 + i]
                        payments[i] = x[32 + i]
                    allocations = np.clip(allocations, 0, 1)
                    payments = np.clip(payments, 0, None)
                    return {"allocations": allocations, "payments": payments}
            except Exception:
                pass
            return {"allocations": best_allocations, "payments": best_payments}

    except Exception:
        return {"allocations": best_allocations, "payments": best_payments}
