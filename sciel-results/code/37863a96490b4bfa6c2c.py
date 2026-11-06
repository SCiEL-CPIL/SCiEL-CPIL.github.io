import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # safety margin before 90s kill

    # Parse distribution_spec to get probabilities over 16 types
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            # Try to extract from keys
            probs = np.zeros(16)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    v1, v2 = k
                    idx = v1 * 4 + v2
                    if 0 <= idx < 16:
                        probs[idx] = v
                elif isinstance(k, int) and 0 <= k < 16:
                    probs[k] = v
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != 16:
            # Try to resize or pad
            if len(probs) > 16:
                probs = probs[:16]
            else:
                tmp = np.zeros(16)
                tmp[:len(probs)] = probs
                probs = tmp
    else:
        probs = np.ones(16) / 16.0

    # Normalize
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(16) / 16.0

    # Ensure non-negative
    probs = np.maximum(probs, 0)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(16) / 16.0

    n_types = 16
    # Values: type i has v1 = i//4, v2 = i%4
    values = np.zeros((n_types, 2))
    for i in range(n_types):
        values[i, 0] = i // 4
        values[i, 1] = i % 4

    # LP formulation:
    # Variables: x[i,j] for i in 0..15, j in 0..1 (allocation probabilities, 0<=x<=1)
    #            p[i] for i in 0..15 (payments, p >= 0)
    # Total: 32 + 16 = 48 variables
    #
    # Objective: Maximize sum_i pi_i * p_i
    # In linprog (minimize): minimize -sum_i pi_i * p_i
    #
    # Constraints:
    # 1. IR: p_i <= sum_j v_ij * x_ij  =>  p_i - sum_j v_ij x_ij <= 0
    # 2. IC: p_i - p_j <= sum_k v_ik(x_ik - x_jk)  =>  p_i - p_j - sum_k v_ik x_ik + sum_k v_ik x_jk <= 0
    #    for all i, j (i != j)
    # 3. Bounds: 0 <= x_ij <= 1, p_i >= 0

    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]  # minimize -revenue

    # Build constraint matrix
    A_ub = []
    b_ub = []

    # IR constraints: p_i - sum_j v_ij x_ij <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        for j in range(2):
            row[i * 2 + j] = -values[i, j]
        A_ub.append(row)
        b_ub.append(0.0)

    # IC constraints: for each pair (i, j), i != j
    # p_i - p_j - sum_k v_ik x_ik + sum_k v_ik x_jk <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            for k in range(2):
                row[i * 2 + k] = -values[i, k]
                row[j * 2 + k] = values[i, k]
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[32:] = 10.0  # payments can go up to max value sum = 6, use 10 as safe bound

    # Solve LP
    best_allocations = None
    best_payments = None
    best_revenue = -np.inf

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
        if res.status == 0 and res.x is not None:
            x_sol = res.x[:32]
            p_sol = res.x[32:]

            # Clean up numerical noise
            x_sol = np.clip(x_sol, 0, 1)
            p_sol = np.maximum(p_sol, 0)

            allocations = x_sol.reshape(16, 2)
            payments = p_sol

            # Verify constraints
            valid = True
            # Check IR
            for i in range(n_types):
                if payments[i] > np.sum(values[i] * allocations[i]) + 1e-6:
                    valid = False
                    break
            # Check IC (spot check a few)
            if valid:
                for i in range(0, n_types, 3):
                    for j in range(0, n_types, 5):
                        if i == j:
                            continue
                        lhs = payments[i] - payments[j]
                        rhs = np.sum(values[i] * (allocations[i] - allocations[j]))
                        if lhs > rhs + 1e-6:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                best_allocations = allocations
                best_payments = payments
                best_revenue = np.sum(probs * payments)
            else:
                # Fallback: clamp payments to satisfy IR
                for i in range(n_types):
                    max_pay = np.sum(values[i] * allocations[i])
                    if payments[i] > max_pay:
                        payments[i] = max_pay
                best_allocations = allocations
                best_payments = payments
                best_revenue = np.sum(probs * payments)

    except Exception:
        pass

    # If LP failed or we're running out of time, use a simple valid fallback
    if best_allocations is None or time.time() > deadline - 5:
        # Simple valid mechanism: allocate nothing, charge nothing
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {"allocations": best_allocations, "payments": best_payments}
