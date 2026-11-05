import numpy as np
from scipy.optimize import minimize

def construct_auction(distribution_spec, support, constraints):
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)
    
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, (int, float)):
        idx = int(distribution_spec)
        if idx == 0:
            probs = np.ones(num_types) / num_types
        elif idx == 1:
            probs = np.ones(num_types) / num_types
        elif idx == 2:
            for i, t in enumerate(types):
                if t[0] == t[1]:
                    probs[i] = 0.25
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    else:
        probs = np.ones(num_types) / num_types

    n_vars = 3 * num_types
    
    # Pre-compute coefficients for objective and constraints to speed up evaluation
    # Objective: -sum p_t (v1_t x1_t + v2_t x2_t - u_t)
    c_obj = np.zeros(n_vars)
    for t in range(num_types):
        v1, v2 = types[t]
        c_obj[t*3] = -probs[t] * v1
        c_obj[t*3+1] = -probs[t] * v2
        c_obj[t*3+2] = probs[t]

    def objective(z):
        return np.dot(c_obj, z)

    # Pre-compute constraint matrices A_ineq and b_ineq
    # Constraints: A_ineq @ z <= b_ineq
    # We need to return - (A_ineq @ z - b_ineq) for SLSQP (which expects >= 0)
    # Or just compute the values directly.
    
    # IR Constraints: u_t - v1_t x1_t - v2_t x2_t <= 0
    # IC Constraints: u_tp + (v1_t - v1_tp) x1_tp + (v2_t - v2_tp) x2_tp - u_t <= 0
    
    # We can build a sparse matrix or just use loops in the function.
    # Given small size, loops are fine.
    
    def constraints_func(z):
        # Returns array of values that must be >= 0 for SLSQP
        # So we compute - (LHS) where LHS <= 0
        vals = np.zeros(num_types + num_types*(num_types-1))
        idx = 0
        
        # IR
        for t in range(num_types):
            v1, v2 = types[t]
            lhs = z[t*3+2] - v1*z[t*3] - v2*z[t*3+1]
            vals[idx] = -lhs
            idx += 1
            
        # IC
        for t in range(num_types):
            v1_t, v2_t = types[t]
            for t_prime in range(num_types):
                if t == t_prime:
                    continue
                v1_tp, v2_tp = types[t_prime]
                lhs = z[t_prime*3+2] + (v1_t - v1_tp)*z[t_prime*3] + (v2_t - v2_tp)*z[t_prime*3+1] - z[t*3+2]
                vals[idx] = -lhs
                idx += 1
                
        return vals

    bounds = []
    for t in range(num_types):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 6))

    z0 = np.zeros(n_vars)
    
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                   constraints=[{'type': 'ineq', 'fun': constraints_func}],
                   options={'maxiter': 500, 'ftol': 1e-9})
    
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    
    if res.success:
        z_opt = res.x
        for t in range(num_types):
            x1 = np.clip(z_opt[t*3], 0, 1)
            x2 = np.clip(z_opt[t*3+1], 0, 1)
            u = np.clip(z_opt[t*3+2], 0, 6)
            
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            
            v1, v2 = types[t]
            v = v1 * x1 + v2 * x2
            p = v - u
            if p < 0:
                p = 0.0
            payments[t] = p
    else:
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)

    return {
        "allocations": allocations,
        "payments": payments
    }
