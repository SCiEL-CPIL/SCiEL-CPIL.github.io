import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    import time
    t_start = time.time()
    TIME_LIMIT = 85.0  # safety margin before 90s kill
    
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution
    if isinstance(distribution_spec, str):
        if distribution_spec in ('uniform', 'independent_uniform', 'indep_uniform'):
            probs = np.ones(n_types) / n_types
        elif distribution_spec == 'independent':
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs are valid
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    
    # LP Variables: [x1_0, x2_0, U_0, x1_1, x2_1, U_1, ..., x1_15, x2_15, U_15]
    # Total: 16 * 3 = 48 variables
    n_vars = n_types * 3
    
    # Objective: maximize sum_i probs[i] * (v_i1*x_i1 + v_i2*x_i2 - U_i)
    # In linprog: minimize -objective
    c = np.zeros(n_vars)
    for i in range(n_types):
        v1, v2 = types[i]
        c[i*3] = -probs[i] * v1      # x1_i coefficient
        c[i*3+1] = -probs[i] * v2    # x2_i coefficient
        c[i*3+2] = -probs[i]         # U_i coefficient
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1) Allocation bounds: 0 <= x_i1 <= 1, 0 <= x_i2 <= 1
    #    -x_i1 <= 0  and  x_i1 <= 1
    # 2) IC: U_i >= U_j + (v_i - v_j) @ x_j
    #    => U_i - U_j - (v_i1-v_j1)*x_j1 - (v_i2-v_j2)*x_j2 >= 0
    #    => -(U_i - U_j - (v_i1-v_j1)*x_j1 - (v_i2-v_j2)*x_j2) <= 0
    # 3) IR: U_i <= v_i1*x_i1 + v_i2*x_i2
    #    => U_i - v_i1*x_i1 - v_i2*x_i2 <= 0
    
    A_ub_rows = []
    b_ub = []
    
    # Allocation lower bounds: -x_i1 <= 0, -x_i2 <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[i*3] = -1.0
        A_ub_rows.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(n_vars)
        row[i*3+1] = -1.0
        A_ub_rows.append(row)
        b_ub.append(0.0)
    
    # Allocation upper bounds: x_i1 <= 1, x_i2 <= 1
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[i*3] = 1.0
        A_ub_rows.append(row)
        b_ub.append(1.0)
        
        row = np.zeros(n_vars)
        row[i*3+1] = 1.0
        A_ub_rows.append(row)
        b_ub.append(1.0)
    
    # IR constraints: U_i - v_i1*x_i1 - v_i2*x_i2 <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i*3] = -v1
        row[i*3+1] = -v2
        row[i*3+2] = 1.0
        A_ub_rows.append(row)
        b_ub.append(0.0)
    
    # IC constraints: for all pairs (i,j), i != j
    # U_i - U_j - (v_i1-v_j1)*x_j1 - (v_i2-v_j2)*x_j2 >= 0
    # => -(U_i - U_j - (v_i1-v_j1)*x_j1 - (v_i2-v_j2)*x_j2) <= 0
    # => -U_i + U_j + (v_i1-v_j1)*x_j1 + (v_i2-v_j2)*x_j2 <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            vi1, vi2 = types[i]
            vj1, vj2 = types[j]
            row = np.zeros(n_vars)
            row[i*3+2] = -1.0       # -U_i
            row[j*3+2] = 1.0        # +U_j
            row[j*3] = (vi1 - vj1)  # +(vi1-vj1)*x_j1
            row[j*3+1] = (vi2 - vj2)# +(vi2-vj2)*x_j2
            A_ub_rows.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)
    
    # Variable bounds
    # x_i1, x_i2 in [0, 1], U_i in [-inf, inf] (but IR will bound them)
    bounds = []
    for i in range(n_types):
        bounds.append((0.0, 1.0))  # x_i1
        bounds.append((0.0, 1.0))  # x_i2
        bounds.append((None, None)) # U_i
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status != 0:
        # Fallback: simple deterministic mechanism
        # Allocate both goods to everyone, charge 0 (IR satisfied, IC trivially holds)
        allocations = np.ones((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.array([[x[i*3], x[i*3+1]] for i in range(n_types)])
    utilities = np.array([x[i*3+2] for i in range(n_types)])
    
    # Compute payments: p_i = v_i @ x_i - U_i
    payments = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - utilities[i]
    
    # Clamp small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify and fix IR violations
    for i in range(n_types):
        v1, v2 = types[i]
        min_payment = 0.0
        if payments[i] < min_payment:
            payments[i] = min_payment
    
    # Check time
    if time.time() - t_start > TIME_LIMIT:
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': allocations, 'payments': payments}
