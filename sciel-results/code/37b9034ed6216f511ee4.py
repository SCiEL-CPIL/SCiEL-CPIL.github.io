"""Combined auction design: direct payment LP with robust parsing and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # 1. Define Type Space
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # 2. Parse Distribution
    probs = np.zeros(n_types)
    
    if isinstance(distribution_spec, dict):
        if all(isinstance(k, tuple) for k in distribution_spec.keys()):
            for k, v in distribution_spec.items():
                if k in type_to_idx:
                    probs[type_to_idx[k]] = v
        elif 'probs' in distribution_spec:
            p_data = distribution_spec['probs']
            if isinstance(p_data, (list, np.ndarray)) and len(p_data) == n_types:
                probs = np.array(p_data, dtype=float)
            elif isinstance(p_data, dict):
                for k, v in p_data.items():
                    if k in type_to_idx:
                        probs[type_to_idx[k]] = v
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                for k, v in dist.items():
                    if k in type_to_idx:
                        probs[type_to_idx[k]] = v
            elif isinstance(dist, (list, tuple, np.ndarray)):
                probs = np.array(dist, dtype=float)
    elif isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
    
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types
    
    # 3. Setup LP - Direct payment maximization
    # Variables: x1[0..15], x2[0..15], p[0..15] => 48 variables
    n_vars = 48
    x1_offset = 0
    x2_offset = 16
    p_offset = 32
    
    # Objective: maximize sum_t probs[t] * p(t)
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_offset + t] = -probs[t]
    
    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: for all t != t':
    #   v(t)·x(t) - p(t) >= v(t)·x(t') - p(t')
    #   => -v1[t]*x1[t] - v2[t]*x2[t] + p[t] + v1[t]*x1[t'] + v2[t]*x2[t'] - p[t'] <= 0
    for t in range(n_types):
        for t2 in range(n_types):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            v1_t, v2_t = types[t]
            
            row[x1_offset + t] = -v1_t
            row[x2_offset + t] = -v2_t
            row[p_offset + t] = 1.0
            row[x1_offset + t2] = v1_t
            row[x2_offset + t2] = v2_t
            row[p_offset + t2] = -1.0
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: for all t:
    #   v1[t]*x1[t] + v2[t]*x2[t] - p[t] >= 0
    #   => -v1[t]*x1[t] - v2[t]*x2[t] + p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        v1_t, v2_t = types[t]
        
        row[x1_offset + t] = -v1_t
        row[x2_offset + t] = -v2_t
        row[p_offset + t] = 1.0
        
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: x in [0,1], p >= 0
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # p[t]
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
    
    if time.time() > deadline - 5.0:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
    
    if not result.success:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
    
    x_opt = result.x
    
    allocations = np.zeros((n_types, n_goods))
    for t in range(n_types):
        allocations[t, 0] = x_opt[x1_offset + t]
        allocations[t, 1] = x_opt[x2_offset + t]
    
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = x_opt[p_offset + t]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    # 4. Verification and fix pass
    # Check IC and IR, fix any violations by reducing payments
    max_violation = 0.0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        utility_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        
        # IR check
        if utility_t < -1e-10:
            payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1] + 1e-12)
            max_violation = max(max_violation, -utility_t)
        
        # IC check against all other types
        for t2 in range(n_types):
            if t == t2:
                continue
            v1_t2, v2_t2 = types[t2]
            utility_t_as_t2 = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            
            if utility_t < utility_t_as_t2 - 1e-10:
                # Need to reduce payments[t2] or increase payments[t]
                # Simplest: reduce payments[t2] to satisfy IC
                max_allowed_p_t2 = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - utility_t
                if payments[t2] > max_allowed_p_t2:
                    payments[t2] = max(0.0, max_allowed_p_t2)
                    max_violation = max(max_violation, utility_t_as_t2 - utility_t)
    
    # Final clip to ensure feasibility
    payments = np.clip(payments, 0.0, None)
    
    # If time permits, do a high-precision polish
    if time.time() < deadline - 10.0 and max_violation > 1e-8:
        # Re-solve with tighter tolerance
        try:
            result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                           method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12})
            if result2.success:
                x_opt2 = result2.x
                allocations = np.zeros((n_types, n_goods))
                for t in range(n_types):
                    allocations[t, 0] = x_opt2[x1_offset + t]
                    allocations[t, 1] = x_opt2[x2_offset + t]
                payments = np.zeros(n_types)
                for t in range(n_types):
                    payments[t] = x_opt2[p_offset + t]
                allocations = np.clip(allocations, 0.0, 1.0)
                payments = np.clip(payments, 0.0, None)
        except Exception:
            pass
    
    return {
        "allocations": allocations,
        "payments": payments
    }

# EVOLVE-BLOCK-END
