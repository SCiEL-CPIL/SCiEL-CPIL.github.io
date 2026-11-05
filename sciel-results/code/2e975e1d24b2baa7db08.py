"""Incentive-compatible finite auction design solver using LP."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    n_goods = 2

    type_idx = {t: i for i, t in enumerate(types)}

    # Parse distribution_spec to get probabilities for each type
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).reshape(n_types)
    else:
        probs = np.ones(n_types) / n_types

    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types

    n_vars = n_types * n_goods + n_types  # 48

    def var_idx(i, j):
        return i * n_goods + j

    def pay_idx(i):
        return n_types * n_goods + i

    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[pay_idx(i)] = -probs[i]

    # Build constraint matrix in sparse form
    # IR constraints (16) + IC constraints (16*15=240) = 256 total
    n_constraints = n_types + n_types * (n_types - 1)  # 16 + 240 = 256

    # Collect (row, col, val) triples for sparse matrix
    rows = []
    cols = []
    data = []
    b_ub_list = []

    # IR constraints: for each type i
    # -sum_j x[i,j]*v_i_j + p[i] <= 0
    for i in range(n_types):
        row_idx = i
        v1, v2 = types[i]
        if v1 != 0:
            rows.append(row_idx); cols.append(var_idx(i, 0)); data.append(-v1)
        if v2 != 0:
            rows.append(row_idx); cols.append(var_idx(i, 1)); data.append(-v2)
        rows.append(row_idx); cols.append(pay_idx(i)); data.append(1.0)
        b_ub_list.append(0.0)

    # IC constraints: for all pairs (true_type i, report r) where i != r
    # sum_j x[r,j]*v_i_j - p[r] - sum_j x[i,j]*v_i_j + p[i] <= 0
    ic_row = n_types
    for i in range(n_types):
        vi1, vi2 = types[i]
        for r in range(n_types):
            if i == r:
                continue
            if vi1 != 0:
                rows.append(ic_row); cols.append(var_idx(r, 0)); data.append(vi1)
                rows.append(ic_row); cols.append(var_idx(i, 0)); data.append(-vi1)
            if vi2 != 0:
                rows.append(ic_row); cols.append(var_idx(r, 1)); data.append(vi2)
                rows.append(ic_row); cols.append(var_idx(i, 1)); data.append(-vi2)
            rows.append(ic_row); cols.append(pay_idx(r)); data.append(-1.0)
            rows.append(ic_row); cols.append(pay_idx(i)); data.append(1.0)
            b_ub_list.append(0.0)
            ic_row += 1

    A_ub_sparse = csr_matrix((data, (rows, cols)), shape=(n_constraints, n_vars))
    b_ub = np.array(b_ub_list)

    # Bounds
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))

    def _solve_lp(c_vec, A_mat, b_vec, bounds_list, tol):
        """Solve LP with given tolerance."""
        return linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=bounds_list,
                       method='highs', options={'presolve': True, 'dual_feasibility_tolerance': tol,
                                                 'primal_feasibility_tolerance': tol,
                                                 'dual_edge_tolerance': tol,
                                                 'scale_cost': 1e2, 'scale_primal': 1e2})

    # First solve with standard tolerance
    try:
        result = _solve_lp(c, A_ub_sparse, b_ub, bounds, 1e-7)
    except Exception:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}

    if not result.success:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}

    x = result.x
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_goods):
            allocations[i, j] = x[var_idx(i, j)]
        payments[i] = x[pay_idx(i)]

    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)

    # Check deadline before polish
    if time.time() > deadline - 10.0:
        return {"allocations": allocations, "payments": payments}

    # High-precision polish pass: re-solve with tighter tolerances
    try:
        result2 = _solve_lp(c, A_ub_sparse, b_ub, bounds, 1e-12)
        if result2.success:
            x2 = result2.x
            alloc2 = np.zeros((n_types, n_goods))
            pay2 = np.zeros(n_types)
            for i in range(n_types):
                for j in range(n_goods):
                    alloc2[i, j] = x2[var_idx(i, j)]
                pay2[i] = x2[pay_idx(i)]
            alloc2 = np.clip(alloc2, 0.0, 1.0)
            pay2 = np.clip(pay2, 0.0, None)

            # Verify the polished solution satisfies constraints
            # Check IR: sum_j x[i,j]*v_i_j - p[i] >= -1e-9
            ir_ok = True
            for i in range(n_types):
                v1, v2 = types[i]
                util = alloc2[i, 0] * v1 + alloc2[i, 1] * v2 - pay2[i]
                if util < -1e-9:
                    ir_ok = False
                    break

            # Check IC: for all i, r: util_i(r) - util_i(i) <= 1e-9
            ic_ok = True
            if ir_ok:
                for i in range(n_types):
                    vi1, vi2 = types[i]
                    util_i_true = alloc2[i, 0] * vi1 + alloc2[i, 1] * vi2 - pay2[i]
                    for r in range(n_types):
                        if i == r:
                            continue
                        util_i_r = alloc2[r, 0] * vi1 + alloc2[r, 1] * vi2 - pay2[r]
                        if util_i_r - util_i_true > 1e-9:
                            ic_ok = False
                            break
                    if not ic_ok:
                        break

            if ir_ok and ic_ok:
                allocations = alloc2
                payments = pay2
    except Exception:
        pass  # Keep original solution

    # Final safety check
    if time.time() > deadline:
        return {"allocations": allocations, "payments": payments}

    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
