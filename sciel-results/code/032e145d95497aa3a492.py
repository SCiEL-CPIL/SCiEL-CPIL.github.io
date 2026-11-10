import time
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    T_LIMIT = 85.0

    types = [(a, b) for a in range(4) for b in range(4)]
    N = 16
    V = np.array([[types[i][0], types[i][1]] for i in range(N)], dtype=np.float64)

    def _elapsed():
        return time.time() - t0

    def _parse_probs():
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                arr = np.asarray(distribution_spec['probs'], dtype=np.float64)
                if arr.shape == (N,):
                    arr = np.maximum(arr, 0.0)
                    s = arr.sum()
                    return arr / s if s > 0 else np.ones(N) / N
            if 'distribution' in distribution_spec:
                d = distribution_spec['distribution']
                if d == 'uniform':
                    return np.ones(N) / N
                if d == 'geometric':
                    base = 0.7
                    arr = np.array([base**types[i][0] * base**types[i][1] for i in range(N)])
                    return arr / arr.sum()
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=np.float64)
            if arr.shape == (N,):
                arr = np.maximum(arr, 0.0)
                s = arr.sum()
                if s > 0:
                    return arr / s
        return np.ones(N) / N

    probs = _parse_probs()

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    # LP Variables: x (N), y (N), p (N)
    # Minimize -sum(pi * p)
    c_obj = np.zeros(3 * N)
    c_obj[2 * N : 3 * N] = -probs

    # Build sparse constraint matrix
    # Variables: x_0..x_15, y_0..y_15, p_0..p_15
    # Total 48 variables

    # IR constraints: p_i <= v1_i * x_i + v2_i * y_i
    # => -v1_i * x_i - v2_i * y_i + p_i <= 0
    # 16 constraints

    # IC constraints: p_i - p_j <= v1_i * (x_i - x_j) + v2_i * (y_i - y_j)
    # => v1_i * x_i - v1_i * x_j + v2_i * y_i - v2_i * y_j + p_i - p_j <= 0
    # 16 * 15 = 240 constraints

    n_constraints = N + N * (N - 1)
    rows = []
    cols = []
    vals = []

    # Add IR constraints
    for i in range(N):
        # -v1_i * x_i
        if V[i, 0] != 0:
            rows.append(i)
            cols.append(i)
            vals.append(-V[i, 0])
        # -v2_i * y_i
        if V[i, 1] != 0:
            rows.append(i)
            cols.append(N + i)
            vals.append(-V[i, 1])
        # +1 * p_i
        rows.append(i)
        cols.append(2 * N + i)
        vals.append(1.0)

    # Add IC constraints
    row_idx = N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # v1_i * x_i
            if V[i, 0] != 0:
                rows.append(row_idx)
                cols.append(i)
                vals.append(V[i, 0])
            # -v1_i * x_j
            if V[i, 0] != 0:
                rows.append(row_idx)
                cols.append(j)
                vals.append(-V[i, 0])
            # v2_i * y_i
            if V[i, 1] != 0:
                rows.append(row_idx)
                cols.append(N + i)
                vals.append(V[i, 1])
            # -v2_i * y_j
            if V[i, 1] != 0:
                rows.append(row_idx)
                cols.append(N + j)
                vals.append(-V[i, 1])
            # +1 * p_i
            rows.append(row_idx)
            cols.append(2 * N + i)
            vals.append(1.0)
            # -1 * p_j
            rows.append(row_idx)
            cols.append(2 * N + j)
            vals.append(-1.0)
            row_idx += 1

    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(n_constraints, 3 * N))
    b_ub = np.zeros(n_constraints)

    # Bounds
    # x, y in [0, 1]
    # p in [0, max_val] where max_val = 3+3=6
    max_payment = 6.0
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, max_payment)] * N

    # Solve LP
    # Use high tolerances to ensure feasibility
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={
                          'presolve': True,
                          'dual_feasibility_tolerance': 1e-12,
                          'primal_feasibility_tolerance': 1e-12,
                          'time_limit': T_LIMIT - _elapsed() - 5.0
                      })
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        # Fallback: Zero mechanism
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    x = np.clip(res.x[:N], 0.0, 1.0)
    y = np.clip(res.x[N:2*N], 0.0, 1.0)
    p = np.maximum(res.x[2*N:3*N], 0.0)

    # Post-process: Ensure strict feasibility with a safety margin
    # The evaluator requires regret <= 1e-8, so we need a margin to account for floating point errors
    SAFETY_MARGIN = 1e-10

    v1 = V[:, 0]
    v2 = V[:, 1]

    # 1. Check IR: p_i <= v1_i*x_i + v2_i*y_i
    util_true = x * v1 + y * v2

    # Reduce payments to satisfy IR with safety margin
    # p_i should be <= util_true_i - SAFETY_MARGIN
    max_p_allowed = util_true - SAFETY_MARGIN
    p = np.minimum(p, max_p_allowed)
    p = np.maximum(p, 0.0)

    # 2. Verify IC: p_i - p_j <= v1_i*(x_i - x_j) + v2_i*(y_i - y_j) for all i, j
    # Since we only reduced payments, and uniform reduction doesn't affect IC slack,
    # but non-uniform reduction (due to IR clipping) might affect IC.
    # Let's check and if violated, we need to fix.

    ic_ok = True
    max_ic_violation = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # p_i - p_j <= v1_i*(x_i - x_j) + v2_i*(y_i - y_j)
            lhs = p[i] - p[j]
            rhs = V[i,0]*(x[i]-x[j]) + V[i,1]*(y[i]-y[j])
            violation = lhs - rhs
            if violation > max_ic_violation:
                max_ic_violation = violation
            if violation > 1e-8:
                ic_ok = False

    if not ic_ok:
        # If IC is violated, we need to adjust payments.
        # One approach: solve a secondary LP to find the best feasible payments.
        # But for simplicity, let's try a conservative approach:
        # Reduce all payments by the max IC violation + margin
        # This preserves IC (uniform shift) and helps IR
        reduction = max_ic_violation + SAFETY_MARGIN
        p = p - reduction
        p = np.maximum(p, 0.0)

        # Re-check IR after reduction
        max_p_allowed = util_true - SAFETY_MARGIN
        p = np.minimum(p, max_p_allowed)
        p = np.maximum(p, 0.0)

        # Final IC check
        ic_ok = True
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                lhs = p[i] - p[j]
                rhs = V[i,0]*(x[i]-x[j]) + V[i,1]*(y[i]-y[j])
                if lhs - rhs > 1e-8:
                    ic_ok = False
                    break
            if not ic_ok:
                break

        if not ic_ok:
            # Last resort: zero mechanism
            return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}

