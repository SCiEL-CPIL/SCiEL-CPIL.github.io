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
    # Order: x0_0, x1_0, u_0, x0_1, x1_1, u_1, ...
    n_vars = 3 * num_types
    
    # Objective function: Maximize Revenue
    # Revenue = sum p_t * (v1_t * x1_t + v2_t * x2_t - u_t)
    # Minimize -Revenue
    def objective(z):
        rev = 0.0
        for t in range(num_types):
            v1, v2 = types[t]
            x1 = z[t*3]
            x2 = z[t*3+1]
            u = z[t*3+2]
            rev += probs[t] * (v1 * x1 + v2 * x2 - u)
        return -rev

    # Gradient of objective
    def objective_grad(z):
        g = np.zeros(n_vars)
        for t in range(num_types):
            v1, v2 = types[t]
            g[t*3] = -probs[t] * v1
            g[t*3+1] = -probs[t] * v2
            g[t*3+2] = probs[t]
        return g

    # Constraints:
    # 1. IR: u_t - v1_t*x1_t - v2_t*x2_t <= 0
    # 2. IC: u_tp + v1_tp*x1_tp + v2_tp*x2_tp - u_t - v1_t*x1_tp - v2_t*x2_tp <= 0
    
    cons = []
    
    # IR Constraints
    for t in range(num_types):
        v1, v2 = types[t]
        def ir_fun(z, t=t):
            return z[t*3+2] - v1*z[t*3] - v2*z[t*3+1]
        
        def ir_grad(z, t=t):
            g = np.zeros(n_vars)
            g[t*3] = -v1
            g[t*3+1] = -v2
            g[t*3+2] = 1.0
            return g
        
        cons.append({'type': 'ineq', 'fun': ir_fun, 'jac': ir_grad})
        
    # IC Constraints
    for t in range(num_types):
        v1_t, v2_t = types[t]
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            v1_tp, v2_tp = types[t_prime]
            
            def ic_fun(z, t=t, t_prime=t_prime, v1_t=v1_t, v2_t=v2_t, v1_tp=v1_tp, v2_tp=v2_tp):
                # u_tp + v_tp . x_tp - u_t - v_t . x_tp <= 0
                # u_tp + v1_tp*x1_tp + v2_tp*x2_tp - u_t - v1_t*x1_tp - v2_t*x2_tp
                return (z[t_prime*3+2] + v1_tp*z[t_prime*3] + v2_tp*z[t_prime*3+1] 
                        - z[t*3+2] - v1_t*z[t_prime*3] - v2_t*z[t_prime*3+1])
            
            def ic_grad(z, t=t, t_prime=t_prime, v1_t=v1_t, v2_t=v2_t, v1_tp=v1_tp, v2_tp=v2_tp):
                g = np.zeros(n_vars)
                # Coeff for x1_tp: v1_tp - v1_t
                g[t_prime*3] = v1_tp - v1_t
                # Coeff for x2_tp: v2_tp - v2_t
                g[t_prime*3+1] = v2_tp - v2_t
                # Coeff for u_tp: 1
                g[t_prime*3+2] = 1.0
                # Coeff for u_t: -1
                g[t*3+2] = -1.0
                return g
            
            cons.append({'type': 'ineq', 'fun': ic_fun, 'jac': ic_grad})

    # Bounds
    bounds = []
    for t in range(num_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
        bounds.append((0, 6)) # u

    # Initial guess
    # Start with x=1, u=v (zero payment, IR holds)
    z0 = np.zeros(n_vars)
    for t in range(num_types):
        v1, v2 = types[t]
        z0[t*3] = 1.0
        z0[t*3+1] = 1.0
        z0[t*3+2] = v1 + v2

    # Solve
    res = minimize(objective, z0, method='SLSQP', jac=objective_grad, 
                   bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
    
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
            # Ensure non-negative payment
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
