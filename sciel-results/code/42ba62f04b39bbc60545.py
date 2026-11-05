import numpy as np
from scipy.optimize import minimize

def construct_auction(distribution_spec, support, constraints):
    # Define the support types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)
    
    # Determine probabilities for each type
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
            # Uniform joint
            probs = np.ones(num_types) / num_types
        elif idx == 1:
            # Independent uniform
            probs = np.ones(num_types) / num_types
        elif idx == 2:
            # Correlated: v1 == v2
            for i, t in enumerate(types):
                if t[0] == t[1]:
                    probs[i] = 0.25
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    else:
        probs = np.ones(num_types) / num_types

    # Variables: x1_t, x2_t, u_t for each type t
    n_vars = 3 * num_types
    
    def get_vars(z, t_idx):
        base = t_idx * 3
        return z[base], z[base + 1], z[base + 2]

    def value(t_idx, x1, x2):
        v1, v2 = types[t_idx]
        return v1 * x1 + v2 * x2

    def revenue(z):
        rev = 0.0
        for t in range(num_types):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            rev += probs[t] * (v - u)
        return rev

    def objective(z):
        return -revenue(z)

    # Constraints
    cons = []
    
    # 1. IR: u_t >= 0
    for t in range(num_types):
        def ir_con(z, t=t):
            _, _, u = get_vars(z, t)
            return u
        cons.append({'type': 'ineq', 'fun': ir_con})
        
    # 2. Non-negative payments: v_t - u_t >= 0
    for t in range(num_types):
        def nn_pay_con(z, t=t):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            return v - u
        cons.append({'type': 'ineq', 'fun': nn_pay_con})

    # 3. IC: u_t - u_tp >= v_t(x_tp) - v_tp(x_tp)
    # => u_t - v_t(x_tp) + v_tp(x_tp) - u_tp >= 0
    for t in range(num_types):
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            def ic_con(z, t=t, t_prime=t_prime):
                x1_t, x2_t, u_t = get_vars(z, t)
                x1_tp, x2_tp, u_tp = get_vars(z, t_prime)
                
                v_t_given_tp = value(t, x1_tp, x2_tp)
                v_tp_given_tp = value(t_prime, x1_tp, x2_tp)
                
                return u_t - v_t_given_tp + v_tp_given_tp - u_tp
            cons.append({'type': 'ineq', 'fun': ic_con})

    bounds = []
    for t in range(num_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
        bounds.append((0, 6)) # u (Max value is 6)

    # Multi-start strategy to avoid local minima/convergence issues
    # Start 1: Zero allocation
    z0_1 = np.zeros(n_vars)
    
    # Start 2: Full allocation, payment = value (Revenue 0, feasible)
    z0_2 = np.zeros(n_vars)
    for t in range(num_types):
        z0_2[t*3] = 1.0
        z0_2[t*3+1] = 1.0
        z0_2[t*3+2] = value(t, 1.0, 1.0)
        
    # Start 3: Threshold heuristic
    # For each good, allocate to types with high value.
    # Let's try allocating good 1 if v1 >= 2, good 2 if v2 >= 2.
    z0_3 = np.zeros(n_vars)
    for t in range(num_types):
        v1, v2 = types[t]
        x1 = 1.0 if v1 >= 2 else 0.0
        x2 = 1.0 if v2 >= 2 else 0.0
        # Set u_t to satisfy IC/IR? 
        # Simplest: set u_t = 0. Check if feasible.
        # If not feasible, SLSQP will adjust.
        z0_3[t*3] = x1
        z0_3[t*3+1] = x2
        z0_3[t*3+2] = 0.0

    starts = [z0_1, z0_2, z0_3]
    
    best_res = None
    best_revenue = -np.inf
    
    for z0 in starts:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 2000, 'ftol': 1e-12})
        
        if res.success:
            # Verify constraints strictly
            valid = True
            # Check bounds
            for i in range(n_vars):
                if res.x[i] < -1e-9 or res.x[i] > bounds[i][1] + 1e-9:
                    valid = False
                    break
            
            if valid:
                # Check constraints
                for c in cons:
                    if c['fun'](res.x) < -1e-8:
                        valid = False
                        break
                        
                if valid:
                    rev = revenue(res.x)
                    if rev > best_revenue:
                        best_revenue = rev
                        best_res = res

    # Extract results
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    
    if best_res is not None:
        z_opt = best_res.x
        for t in range(num_types):
            x1, x2, u = get_vars(z_opt, t)
            # Clamp x to [0,1]
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            
            v = value(t, x1, x2)
            p = v - u
            # Clamp payment to be non-negative
            if p < 0:
                p = 0.0
            payments[t] = p
    else:
        # Fallback: No allocation
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)

    return {
        "allocations": allocations,
        "payments": payments
    }
