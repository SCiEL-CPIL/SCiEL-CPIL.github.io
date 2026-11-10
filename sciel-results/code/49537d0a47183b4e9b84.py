import time
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    T_LIMIT = 85.0
    
    # 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
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
        return _fallback_mechanism()

    # LP Variables: [x_0..x_15, y_0..y_15, p_0..p_15]
    # Total 48 variables
    n_vars = 3 * N
    
    # Objective: Minimize -sum(probs * p)
    c_obj = np.zeros(n_vars)
    c_obj[2 * N:3 * N] = -probs

    # Build Sparse Constraints
    # Rows: IR (N), IC (N*(N-1))
    n_ic = N * (N - 1)
    total_rows = N + n_ic
    
    # IR: -v1_i*x_i - v2_i*y_i + p_i <= 0
    # IC: v1_i*x_j - v1_i*x_i + v2_i*y_j - v2_i*y_i - p_j + p_i <= 0
    
    # Using COO format for sparse matrix
    row_idx = []
    col_idx = []
    data = []
    b_ub = []
    
    # IR Constraints
    for i in range(N):
        r = i
        b_ub.append(0.0)
        if V[i, 0] != 0:
            row_idx.append(r)
            col_idx.append(i)
            data.append(-V[i, 0])
        if V[i, 1] != 0:
            row_idx.append(r)
            col_idx.append(N + i)
            data.append(-V[i, 1])
        row_idx.append(r)
        col_idx.append(2 * N + i)
        data.append(1.0)
        
    # IC Constraints
    idx = N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = idx
            b_ub.append(0.0)
            # v_i . (a_j - a_i) - (p_j - p_i) <= 0
            # v1_i*x_j - v1_i*x_i + v2_i*y_j - v2_i*y_i - p_j + p_i <= 0
            
            if V[i, 0] != 0:
                row_idx.append(r)
                col_idx.append(j)
                data.append(V[i, 0])
                row_idx.append(r)
                col_idx.append(i)
                data.append(-V[i, 0])
            if V[i, 1] != 0:
                row_idx.append(r)
                col_idx.append(N + j)
                data.append(V[i, 1])
                row_idx.append(r)
                col_idx.append(N + i)
                data.append(-V[i, 1])
                
            row_idx.append(r)
            col_idx.append(2 * N + j)
            data.append(-1.0)
            
            row_idx.append(r)
            col_idx.append(2 * N + i)
            data.append(1.0)
            
            idx += 1
            
    A_ub_sparse = sp.coo_matrix((data, (row_idx, col_idx)), shape=(total_rows, n_vars)).tocsr()
    b_ub = np.array(b_ub)
    
    # Bounds
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, None)] * N
    
    if _elapsed() > T_LIMIT:
        return _fallback_mechanism()

    res = None
    try:
        # Use HiGHS with sparse matrix
        res = linprog(c_obj, A_ub=A_ub_sparse, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12, 'time_limit': 40.0})
    except Exception:
        pass

    if res and res.success and res.x is not None:
        x = np.clip(res.x[:N], 0.0, 1.0)
        y = np.clip(res.x[N:2 * N], 0.0, 1.0)
        p = np.maximum(res.x[2 * N:3 * N], 0.0)
        
        # Refine Payments (Secondary LP) to fix numerical drift
        if _elapsed() < T_LIMIT - 10.0:
            p = _refine_payments(x, y, V, probs)
            
        # Final Verification with Slack
        if _is_feasible(x, y, p, V, tol=1e-7):
            return {'allocations': np.column_stack((x, y)), 'payments': p}
        else:
            # Try to fix by reducing payments slightly
            p_fixed = np.maximum(p - 1e-6, 0.0)
            if _is_feasible(x, y, p_fixed, V, tol=1e-7):
                return {'allocations': np.column_stack((x, y)), 'payments': p_fixed}
            # If still infeasible, fallback
            return _fallback_mechanism()
    else:
        return _fallback_mechanism()

def _fallback_mechanism():
    # Separable Threshold Mechanism
    # For each good k: if v_k >= 2, allocate 1, pay 2. Else allocate 0, pay 0.
    types = [(a, b) for a in range(4) for b in range(4)]
    N = 16
    allocations = np.zeros((N, 2))
    payments = np.zeros(N)
    for i in range(N):
        v1, v2 = types[i]
        if v1 >= 2:
            allocations[i, 0] = 1.0
            payments[i] += 2.0
        if v2 >= 2:
            allocations[i, 1] = 1.0
            payments[i] += 2.0
    return {'allocations': allocations, 'payments': payments}

def _refine_payments(x, y, V, probs):
    N = len(x)
    n_ic = N * (N - 1)
    
    # Objective: Max sum(probs * p) -> Min -sum(probs * p)
    c_p = -probs
    
    # Constraints on p only
    # IR: p_i <= V_i . a_i
    # IC: p_i - p_j <= V_i . (a_i - a_j)
    
    row_idx = []
    col_idx = []
    data = []
    b_ub = []
    
    # IR
    for i in range(N):
        r = i
        max_pay = x[i]*V[i,0] + y[i]*V[i,1]
        b_ub.append(max_pay)
        row_idx.append(r)
        col_idx.append(i)
        data.append(1.0)
        
    # IC
    idx = N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = idx
            val_i = x[i]*V[i,0] + y[i]*V[i,1]
            val_j = x[j]*V[i,0] + y[j]*V[i,1]
            b_ub.append(val_i - val_j)
            
            row_idx.append(r)
            col_idx.append(i)
            data.append(1.0)
            
            row_idx.append(r)
            col_idx.append(j)
            data.append(-1.0)
            
            idx += 1
            
    A_p_sparse = sp.coo_matrix((data, (row_idx, col_idx)), shape=(len(b_ub), N)).tocsr()
    b_ub = np.array(b_ub)
    
    bounds_p = [(0.0, None)] * N
    
    try:
        res_p = linprog(c_p, A_ub=A_p_sparse, b_ub=b_ub, bounds=bounds_p, method='highs',
                        options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                 'primal_feasibility_tolerance': 1e-12, 'time_limit': 10.0})
        if res_p.success and res_p.x is not None:
            return np.maximum(res_p.x, 0.0)
    except Exception:
        pass
    return None

def _is_feasible(x, y, p, V, tol=1e-7):
    N = len(x)
    for i in range(N):
        # IR
        util = x[i]*V[i,0] + y[i]*V[i,1] - p[i]
        if util < -tol:
            return False
        # IC
        for j in range(N):
            if i == j: continue
            ur = x[j]*V[i,0] + y[j]*V[i,1] - p[j]
            ut = util
            if ur - ut > tol:
                return False
    return True
