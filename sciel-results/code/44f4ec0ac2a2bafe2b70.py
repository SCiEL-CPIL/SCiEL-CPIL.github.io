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
            if (0, 0) in distribution_spec:
                probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
            elif 0 in distribution_spec:
                probs = np.array([distribution_spec.get(i, 0.0) for i in range(n_types)], dtype=float)
            else:
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

    # LP Formulation with reduced IC constraints
    # Variables:
    # 0-31: x_{v,1}, x_{v,2} for v in 0..15
    # 32-47: p_v for v in 0..15
    n_alloc = n_types * 2
    n_pay = n_types
    n_vars = n_alloc + n_pay
    
    def alloc_idx(v_idx, g):
        return v_idx * 2 + g
        
    def pay_idx(v_idx):
        return n_alloc + v_idx

    # Objective: Maximize sum pi_v * p_v
    c = np.zeros(n_vars)
    for v_idx in range(n_types):
        c[pay_idx(v_idx)] = -probs[v_idx]
        
    # Constraints
    A_ub = []
    b_ub = []
    
    # 1. IC Constraints (reduced): Only check "immediate" neighbors in the type lattice
    # For 2D types (v1,v2), monotonicity requires:
    # x(v1+1,v2) >= x(v1,v2) and x(v1,v2+1) >= x(v1,v2) component-wise
    # We add IC constraints only for adjacent types (differing by 1 in one coordinate)
    
    # Build adjacency: for each type, find types that are "above" it by one step
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        
        # Check if we can increase v1 by 1
        if v1 < 3:
            vp = (v1 + 1, v2)
            vp_idx = type_to_idx[vp]
            # IC: u(v, v) >= u(v, v') where v' = (v1+1, v2)
            # v1*x[v,0] + v2*x[v,1] - p[v] >= v1*x[v',0] + v2*x[v',1] - p[v']
            row = np.zeros(n_vars)
            row[alloc_idx(v_idx, 0)] = -v1
            row[alloc_idx(v_idx, 1)] = -v2
            row[pay_idx(v_idx)] = 1.0
            row[alloc_idx(vp_idx, 0)] = v1
            row[alloc_idx(vp_idx, 1)] = v2
            row[pay_idx(vp_idx)] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
        
        # Check if we can increase v2 by 1
        if v2 < 3:
            vp = (v1, v2 + 1)
            vp_idx = type_to_idx[vp]
            row = np.zeros(n_vars)
            row[alloc_idx(v_idx, 0)] = -v1
            row[alloc_idx(v_idx, 1)] = -v2
            row[pay_idx(v_idx)] = 1.0
            row[alloc_idx(vp_idx, 0)] = v1
            row[alloc_idx(vp_idx, 1)] = v2
            row[pay_idx(vp_idx)] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # 2. IR Constraints:
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
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_alloc = result.x[:n_alloc].reshape(n_types, 2)
            p_pay = result.x[n_alloc:n_alloc+n_pay]
            
            x_alloc = np.clip(x_alloc, 0, 1)
            p_pay = np.clip(p_pay, 0, None)
            
            # Verify full IC (all pairs) since we only enforced adjacent ones
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
                return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
        else:
            return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
    except Exception:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
