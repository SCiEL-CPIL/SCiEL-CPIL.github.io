import time
import numpy as np
from scipy.optimize import linprog

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

    # LP variables: [x_0..x_15, y_0..y_15, p_0..p_15]
    c_obj = np.zeros(3 * N)
    c_obj[2 * N:3 * N] = -probs  # minimize negative revenue

    # IR: for each i, p[i] <= x[i]*v1_i + y[i]*v2_i
    # => -v1_i*x[i] - v2_i*y[i] + p[i] <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -V[i, 0]
        A_ir[i, N + i] = -V[i, 1]
        A_ir[i, 2 * N + i] = 1.0
    b_ir = np.zeros(N)

    # IC: for each (i,j), i!=j:
    # (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    # Build as matrix
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A_ic[idx, j] = V[i, 0]
            A_ic[idx, i] = -V[i, 0]
            A_ic[idx, N + j] = V[i, 1]
            A_ic[idx, N + i] = -V[i, 1]
            A_ic[idx, 2 * N + j] = -1.0
            A_ic[idx, 2 * N + i] = 1.0
            idx += 1
    b_ic = np.zeros(n_ic)

    A_ub = np.vstack([A_ir, A_ic])
    b_ub = np.concatenate([b_ir, b_ic])

    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, None)] * N

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    x = np.clip(res.x[:N], 0.0, 1.0)
    y = np.clip(res.x[N:2 * N], 0.0, 1.0)
    p = np.maximum(res.x[2 * N:3 * N], 0.0)

    # Post-process: enforce IR by capping payments
    v1 = V[:, 0]
    v2 = V[:, 1]
    p_cap = x * v1 + y * v2
    p = np.minimum(p, p_cap)
    p = np.maximum(p, 0.0)

    # Verify IC; if violated, reduce payments on types that cause violations
    if _elapsed() < T_LIMIT:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                util_rep = x[j] * V[i, 0] + y[j] * V[i, 1] - p[j]
                util_true = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
                if util_rep - util_true > 1e-8:
                    # Increase p[j] or decrease p[i] to fix
                    deficit = util_rep - util_true + 1e-9
                    p[i] = max(p[i] - deficit, 0.0)
                    # Re-check after adjustment
                    util_true_new = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
                    if util_true_new < 0:
                        p[i] = x[i] * V[i, 0] + y[i] * V[i, 1]

    # Final verification
    ic_ok = True
    for i in range(N):
        if x[i] * V[i, 0] + y[i] * V[i, 1] - p[i] < -1e-8:
            ic_ok = False
            break
        for j in range(N):
            if i == j:
                continue
            ur = x[j] * V[i, 0] + y[j] * V[i, 1] - p[j]
            ut = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
            if ur - ut > 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}
