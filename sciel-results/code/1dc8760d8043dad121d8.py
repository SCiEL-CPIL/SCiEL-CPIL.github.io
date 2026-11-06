# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, with 16 types.
    
    Returns:
        allocations: (16, 2) array of allocation probabilities
        payments: (16,) array of expected payments
    """
    t_start = time.time()
    deadline = t_start + 85.0
    
    # Parse the 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index: idx = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.array(probs, dtype=float).flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    values = np.array([[v1, v2] for v1, v2 in types], dtype=float)
    
    # For a single additive bidder, incentive compatibility implies monotone allocations
    # and the payment is determined by the allocation rule (via the integral formula).
    # We can reformulate: maximize expected revenue subject to monotonicity and IR.
    # 
    # For type t with values v_t = (v1_t, v2_t), the utility is u_t = v_t · x_t - p_t.
    # IC: u_t >= v_t · x_{t'} - p_{t'} for all t, t'
    # This is equivalent to: u_t - u_{t'} >= v_t · (x_t - x_{t'})
    # 
    # With IR: u_t >= 0, and the lowest type (0,0) has u_{0,0} = 0 (binding).
    # 
    # We can use the standard result: for a single bidder, the optimal mechanism
    # maximizes expected virtual revenue. The virtual value for type t is:
    # phi_t = v_t - (1 - F_t)/f_t * dv/dt  (in continuous case)
    # In discrete case, we can use the LP formulation.
    
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Build constraints
    # x[t,g] in [0, 1]
    # p[t] >= 0
    # IR: p[t] - sum_g x[t,g]*v[t,g] <= 0  =>  sum_g x[t,g]*v[t,g] - p[t] >= 0
    # IC: for all t, t': 
    #   (v_t · x_t - p_t) >= (v_t · x_{t'} - p_{t'})
    #   => v_t · (x_t - x_{t'}) - (p_t - p_{t'}) >= 0
    
    n_constraints = n_x + n_p + n_types * n_types
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    # x[t,g] <= 1
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = 1.0
            b_ub[row] = 1.0
            row += 1
    
    # IR: p[t] - sum_g x[t,g]*v[t,g] <= 0
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = -values[t, g]
        A_ub[row, n_x + t] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # IC: for all t, t' (t != t')
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            for g in range(2):
                A_ub[row, t_prime * 2 + g] = values[t, g]
                A_ub[row, t * 2 + g] = -values[t, g]
            A_ub[row, n_x + t] = 1.0
            A_ub[row, n_x + t_prime] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    # Adjust size
    A_ub = A_ub[:row, :]
    b_ub = b_ub[:row]
    
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    
    # Solve with tighter tolerances
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                              'primal_feasibility_tolerance': 1e-10,
                              'time_limit': 80.0})
    
    if time.time() > deadline - 2.0:
        # Time's up, return a feasible fallback
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if result.status != 0:
        # Fallback: trivially feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_flat = result.x[:n_x].reshape(n_types, 2)
    p = result.x[n_x:]
    
    # Clip with small margin to ensure strict feasibility
    eps = 1e-10
    x_flat = np.clip(x_flat, eps, 1.0 - eps)
    p = np.clip(p, eps, None)
    
    # Verify feasibility and fix if needed
    # Check IR
    for t in range(n_types):
        utility = values[t] @ x_flat[t] - p[t]
        if utility < -1e-8:
            p[t] = values[t] @ x_flat[t] + 1e-9
    
    # Check IC
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            utility_t = values[t] @ x_flat[t] - p[t]
            utility_t_prime = values[t] @ x_flat[t_prime] - p[t_prime]
            if utility_t < utility_t_prime - 1e-8:
                # Violation: adjust payment
                p[t] = values[t] @ x_flat[t] - utility_t_prime + 1e-9
                p[t] = max(p[t], 0)
    
    return {'allocations': x_flat, 'payments': p}
# EVOLVE-BLOCK-END
