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

    # Stage 1: Full LP to determine optimal allocations and initial payments
    # Variables: x[0..N-1], y[0..N-1], p[0..N-1]
    c_obj = np.zeros(3 * N)
    c_obj[2 * N:3 * N] = -probs

    # IR: p[i] - x[i]*v1[i] - y[i]*v2[i] <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -V[i, 0]
        A_ir[i, N + i] = -V[i, 1]
        A_ir[i, 2 * N + i] = 1.0
    b_ir = np.zeros(N)

    # IC: (x[j]-x[i])*v1[i] + (y[j]-y[i])*v2[i] - p[j] + p[i] <= 0
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

    max_payment = np.array([V[i, 0] + V[i, 1] for i in range(N)])
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, max_payment[i]) for i in range(N)]

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
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

    # Stage 2: Refine payments with a smaller LP, fixing x and y
    # This allows the solver to find the exact optimal payment vector for the
    # fixed allocation rule, potentially recovering revenue lost to numerical
    # slack or suboptimal payment choices in the first stage.
    
    if _elapsed() < T_LIMIT - 10.0:
        # Variables: p[0..N-1]
        c_p = -probs.copy()
        
        # IR: p[i] <= x[i]*v1[i] + y[i]*v2[i]
        # p[i] - 0 <= x[i]*v1[i] + y[i]*v2[i]
        A_ir_p = np.eye(N)
        b_ir_p = x * v1 + y * v2
        
        # IC: (x[j]-x[i])*v1[i] + (y[j]-y[i])*v2[i] - p[j] + p[i] <= 0
        # p[i] - p[j] <= -((x[j]-x[i])*v1[i] + (y[j]-y[i])*v2[i])
        # Let's rewrite: p[i] - p[j] <= x[i]*v1[i] + y[i]*v2[i] - x[j]*v1[i] - y[j]*v2[i]
        # This is the same as before, but x, y are constants now.
        
        A_ic_p = np.zeros((n_ic, N))
        b_ic_p = np.zeros(n_ic)
        idx = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # p[i] - p[j] <= RHS
                A_ic_p[idx, i] = 1.0
                A_ic_p[idx, j] = -1.0
                rhs = (x[i] - x[j]) * V[i, 0] + (y[i] - y[j]) * V[i, 1]
                b_ic_p[idx] = rhs
                idx += 1
        
        A_ub_p = np.vstack([A_ir_p, A_ic_p])
        b_ub_p = np.concatenate([b_ir_p, b_ic_p])
        
        bounds_p = [(0.0, p_cap[i]) for i in range(N)]
        
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                     'primal_feasibility_tolerance': 1e-14})
            if res_p.success and res_p.x is not None:
                p_new = np.maximum(res_p.x, 0.0)
                # Ensure IR is strictly satisfied
                p_new = np.minimum(p_new, p_cap)
                p_new = np.maximum(p_new, 0.0)
                
                # Check if new p is better
                if np.sum(p_new * probs) > np.sum(p * probs) - 1e-12:
                    p = p_new
        except Exception:
            pass

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
