import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Reformulate the mechanism design LP using utility variables (u_i) 
    instead of payments (p_i). This keeps variables bounded and improves 
    numerical stability. Solve with HiGHS.
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
    # Variables: x1 (16), x2 (16), u (16)
    n_vars = 48

    # 3. Formulate LP with Utility Variables
    # Maximize Revenue = sum(p_i * probs_i)
    # p_i = v_i . x_i - u_i
    # Revenue = sum( (v_i . x_i - u_i) * probs_i )
    # = sum( (v1_i * x1_i + v2_i * x2_i) * probs_i ) - sum( u_i * probs_i )
    # Minimize -Revenue
    
    c = np.zeros(n_vars)
    for i in range(16):
        v1, v2 = grid_values[i]
        c[i] = -probs[i] * v1
        c[16 + i] = -probs[i] * v2
        c[32 + i] = probs[i] # Minimize -(-u_i * probs_i) => +u_i * probs_i

    A_rows = []
    b_rows = []

    # IR Constraints: u_i >= 0
    # This is handled by bounds on u_i (lower bound 0)
    
    # IC Constraints: u_i >= u_j + (v_i - v_j) . x_j
    # u_i - u_j - (v1_i - v1_j)*x1_j - (v2_i - v2_j)*x2_j >= 0
    # => -u_i + u_j + (v1_i - v1_j)*x1_j + (v2_i - v2_j)*x2_j <= 0
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = grid_values[j]
            row = np.zeros(n_vars)
            # -u_i
            row[32 + i] = -1
            # +u_j
            row[32 + j] = 1
            # +(v1_i - v1_j) * x1_j
            row[j] = v1_i - v1_j
            # +(v2_i - v2_j) * x2_j
            row[16 + j] = v2_i - v2_j
            A_rows.append(row)
            b_rows.append(0)

    # Payment Non-negativity: p_i >= 0 => v_i . x_i - u_i >= 0
    # u_i <= v1_i * x1_i + v2_i * x2_i
    # u_i - v1_i * x1_i - v2_i * x2_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[32 + i] = 1
        row[i] = -v1
        row[16 + i] = -v2
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds:
    # x in [0, 1]
    # u in [0, inf) -- though effectively bounded by v.x
    bounds = [(0, 1)] * 32 + [(0, None)] * 16

    # 4. Solve with HiGHS
    from scipy.optimize import linprog

    def extract_solution(x):
        """Extract allocations and payments from LP variable vector."""
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        utilities = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[i]
            allocations[i, 1] = x[16 + i]
            utilities[i] = x[32 + i]
            v1, v2 = grid_values[i]
            payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - utilities[i]
        
        allocations = np.clip(allocations, 0, 1)
        # Payments should be non-negative, but numerical error might make them slightly negative
        payments = np.clip(payments, 0, None)
        return allocations, payments

    def check_feasibility(allocations, payments, probs):
        """Check if the solution satisfies all constraints within tolerance."""
        # Check IR
        for i in range(16):
            v1, v2 = grid_values[i]
            utility = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
            if utility < -1e-10:
                return False
        # Check IC
        for i in range(16):
            v1_i, v2_i = grid_values[i]
            for j in range(16):
                if i == j:
                    continue
                u_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
                u_j = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
                if u_i < u_j - 1e-10:
                    return False
        # Check bounds
        if np.any(allocations < -1e-12) or np.any(allocations > 1 + 1e-12):
            return False
        if np.any(payments < -1e-12):
            return False
        return True

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})

        if time.time() - start_time > DEADLINE:
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

        if res.success:
            x = res.x
            allocations, payments = extract_solution(x)
            
            # Verify feasibility
            if not check_feasibility(allocations, payments, probs):
                # Try tighter tolerance solve
                try:
                    res_tight = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                       options={'presolve': True,
                                                'dual_feasibility_tolerance': 1e-14,
                                                'primal_feasibility_tolerance': 1e-14})
                    if res_tight.success:
                        x_tight = res_tight.x
                        alloc_tight, pay_tight = extract_solution(x_tight)
                        if check_feasibility(alloc_tight, pay_tight, probs):
                            allocations = alloc_tight
                            payments = pay_tight
                except Exception:
                    pass
            
            # Polish pass
            if time.time() - start_time < DEADLINE - 10.0:
                try:
                    res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-16,
                                            'primal_feasibility_tolerance': 1e-16,
                                            'time_limit': 30.0})
                    if res2.success:
                        x2 = res2.x
                        alloc2, pay2 = extract_solution(x2)
                        if check_feasibility(alloc2, pay2, probs):
                            rev1 = np.sum(payments * probs)
                            rev2 = np.sum(pay2 * probs)
                            if rev2 > rev1 - 1e-12:
                                allocations = alloc2
                                payments = pay2
                except Exception:
                    pass

            return {"allocations": allocations, "payments": payments}
        else:
            # Fallback: try without presolve
            try:
                res_fb = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                 options={'presolve': False,
                                          'dual_feasibility_tolerance': 1e-12,
                                          'primal_feasibility_tolerance': 1e-12})
                if res_fb.success:
                    x = res_fb.x
                    allocations, payments = extract_solution(x)
                    if check_feasibility(allocations, payments, probs):
                        return {"allocations": allocations, "payments": payments}
            except Exception:
                pass
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
