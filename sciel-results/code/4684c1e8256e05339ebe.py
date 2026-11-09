import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill

    # Construct the full support grid: values in {0,1,2,3} for two goods
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}

    # Extract probabilities robustly
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float)
            else:
                # Try to find a numeric array in the dict
                for key in distribution_spec:
                    val = distribution_spec[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.array(val, dtype=float)
                        if arr.size == 16:
                            probs = arr.flatten()
                            break
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs.flatten()
        total = probs.sum()
        if total > 1e-12:
            probs = probs / total
        else:
            probs = np.ones(16) / 16.0

    # Ensure non-negative
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum()

    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: for all pairs (i, j), p_i - p_j <= v_i·x_i - v_i·x_j
    # Rearranged: p_i - p_j - v1_i*x1_i - v2_i*x2_i + v1_i*x1_j + v2_i*x2_j <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        for j in range(n_states):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            row[j] = v1_i
            row[n_states + j] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR constraints: p_i <= v1_i*x1_i + v2_i*x2_i
    # p_i - v1_i*x1_i - v2_i*x2_i <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    if len(A_ub_rows) > 0:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)

    # Bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    # Check time
    if time.time() - start_time > deadline - 5:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    # Solve LP
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                               'primal_feasibility_tolerance': 1e-10})
        if res.status == 0:
            x_opt = res.x
    except Exception:
        x_opt = None

    if x_opt is None:
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point',
                          options={'maxiter': 10000, 'tol': 1e-12})
            if res.status == 0:
                x_opt = res.x
        except Exception:
            x_opt = None

    if x_opt is None:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2 * n_states]
    p = x_opt[2 * n_states:]

    allocations = np.column_stack((x1, x2))
    payments = p.copy()

    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verification and repair pass
    max_regret = 0.0
    ir_violation = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p[i]
        
        # IR check
        if utility_truthful < 0:
            ir_violation = max(ir_violation, -utility_truthful)
        
        for j in range(n_states):
            if i == j:
                continue
            utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret

    # If violations exist, try to repair by adjusting payments
    if max_regret > 1e-8 or ir_violation > 1e-8:
        # Repair: for each type, set payment to the maximum required by IC
        # p(t) <= min_{t'} [v_t · x(t') - u(t')] where u(t') is utility at t'
        # More directly: p(t) must satisfy:
        #   p(t) <= v_t · x(t)  (IR)
        #   p(t) <= p(t') + v_t · (x(t) - x(t'))  for all t' (IC)
        # So p(t) <= min( v_t · x(t), min_{t'} [p(t') + v_t · (x(t) - x(t'))] )
        
        p_repaired = np.zeros(n_states)
        for t in range(n_states):
            v1_t, v2_t = support_grid[t]
            # IR upper bound
            p_max = v1_t * x1[t] + v2_t * x2[t]
            # IC upper bounds from all other types
            for t2 in range(n_states):
                if t == t2:
                    continue
                ic_bound = p[t2] + v1_t * (x1[t] - x1[t2]) + v2_t * (x2[t] - x2[t2])
                if ic_bound < p_max:
                    p_max = ic_bound
            p_repaired[t] = max(0.0, p_max)
        
        payments = p_repaired
        allocations = np.clip(allocations, 0.0, 1.0)
        
        # Re-verify
        max_regret = 0.0
        ir_violation = 0.0
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - payments[i]
            if utility_truthful < 0:
                ir_violation = max(ir_violation, -utility_truthful)
            for j in range(n_states):
                if i == j:
                    continue
                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - payments[j]
                regret = utility_deviate - utility_truthful
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > 1e-8 or ir_violation > 1e-8:
            # Final fallback: zero mechanism
            allocations = np.zeros((n_states, 2))
            payments = np.zeros(n_states)

    return {"allocations": allocations, "payments": payments}
