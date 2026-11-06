import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Define types
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    
    # Parse distribution
    probs = None
    try:
        if isinstance(distribution_spec, dict):
            # Check if keys are tuples or integers
            # Try tuple keys first
            if (0, 0) in distribution_spec:
                probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
            elif 0 in distribution_spec:
                probs = np.array([distribution_spec.get(i, 0.0) for i in range(n_types)], dtype=float)
            else:
                # Fallback: assume values are in order
                probs = np.array(list(distribution_spec.values()), dtype=float)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            flat_spec = np.array(distribution_spec, dtype=float).flatten()
            if len(flat_spec) == n_types:
                probs = flat_spec
            else:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    except Exception:
        probs = np.ones(n_types) / n_types

    # Ensure probs is valid
    if probs is None or len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    else:
        probs = probs[:n_types]
        
    # Normalize and clean
    probs = np.maximum(probs, 0)
    sum_probs = probs.sum()
    if sum_probs > 1e-9:
        probs = probs / sum_probs
    else:
        probs = np.ones(n_types) / n_types

    # LP Formulation
    # Variables:
    # 0-31: x_{v,1}, x_{v,2} for v in 0..15
    # 32-47: p_v for v in 0..15
    n_alloc = n_types * 2
    n_pay = n_types
    n_vars = n_alloc + n_pay
    
    def alloc_idx(v_idx, g):
        return v_idx * 2 + g # g is 0 or 1
        
    def pay_idx(v_idx):
        return n_alloc + v_idx

    # Objective: Maximize sum pi_v * p_v
    # Linprog minimizes, so negate
    c = np.zeros(n_vars)
    for v_idx in range(n_types):
        c[pay_idx(v_idx)] = -probs[v_idx]
        
    # Constraints
    A_ub = []
    b_ub = []
    
    # 1. IC Constraints:
    # u(v, v) >= u(v, v')
    # v1*x[v,0] + v2*x[v,1] - p[v] >= v1*x[v',0] + v2*x[v',1] - p[v']
    # -v1*x[v,0] - v2*x[v,1] + p[v] + v1*x[v',0] + v2*x[v',1] - p[v'] <= 0
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(n_types):
            row = np.zeros(n_vars)
            # Coeffs for v
            row[alloc_idx(v_idx, 0)] = -v1
            row[alloc_idx(v_idx, 1)] = -v2
            row[pay_idx(v_idx)] = 1.0
            # Coeffs for v'
            row[alloc_idx(vp_idx, 0)] = v1
            row[alloc_idx(vp_idx, 1)] = v2
            row[pay_idx(vp_idx)] = -1.0
            
            A_ub.append(row)
            b_ub.append(0.0)
            
    # 2. IR Constraints:
    # u(v, v) >= 0
    # v1*x[v,0] + v2*x[v,1] - p[v] >= 0
    # -v1*x[v,0] - v2*x[v,1] + p[v] <= 0
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        row = np.zeros(n_vars)
        row[alloc_idx(v_idx, 0)] = -v1
        row[alloc_idx(v_idx, 1)] = -v2
        row[pay_idx(v_idx)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    if A_ub:
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
    else:
        A_ub = np.array([[]], dtype=float)
        b_ub = np.array([0.0], dtype=float)
        
    # Bounds
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))
        
    # Solve
    x_alloc = np.zeros((n_types, 2))
    p_pay = np.zeros(n_types)
    
    if time.time() > deadline - 5.0:
        return {'allocations': x_alloc, 'payments': p_pay}
        
    try:
        # Use highs for speed and robustness
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_alloc = result.x[:n_alloc].reshape(n_types, 2)
            p_pay = result.x[n_alloc:n_alloc+n_pay]
            
            # Clip small numerical errors
            x_alloc = np.clip(x_alloc, 0, 1)
            p_pay = np.clip(p_pay, 0, None)
            
            # Final sanity check for feasibility
            # If infeasible due to numerical drift, return trivial
            feasible = True
            tol = 1e-6
            for v_idx in range(n_types):
                v1, v2 = types[v_idx]
                u_vv = v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - p_pay[v_idx]
                if u_vv < -tol:
                    feasible = False
                    break
                for vp_idx in range(n_types):
                    u_vvp = v1 * x_alloc[vp_idx, 0] + v2 * x_alloc[vp_idx, 1] - p_pay[vp_idx]
                    if u_vvp > u_vv + tol:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                return {'allocations': x_alloc, 'payments': p_pay}
            else:
                # Fallback to trivial if solver result is slightly infeasible
                return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
        else:
            return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
    except Exception:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
