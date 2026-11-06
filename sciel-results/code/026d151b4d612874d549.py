"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Parse inputs
    # distribution_spec: likely a list of 16 probabilities or a dict mapping report->prob
    # support: list of 16 tuples (v1, v2)
    # constraints: dict with various constraint parameters
    
    # Extract distribution
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(s, 0.0) for s in support], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            pass
        else:
            # Try to construct from other formats
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0
    
    # Extract support
    if isinstance(support, (list, tuple, np.ndarray)):
        if len(support) == 16:
            support_list = [tuple(s) for s in support]
        else:
            support_list = [(i, j) for i in range(4) for j in range(4)]
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    # Extract constraints
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        regret_bound = constraints.get('regret_bound', constraints.get('max_regret', 1e-8))
    
    n = 16  # number of reports
    m = 2   # number of goods
    
    # Create value matrix: values[i] = (v1_i, v2_i)
    values = np.array(support_list, dtype=float)  # shape (16, 2)
    
    # We need to find allocations x[i,j] (probability of allocating each good to bidder i)
    # and payments p[i]
    # 
    # Variables: x[0..15, 0..1] (32 vars), p[0..15] (16 vars)
    # Total: 48 variables
    #
    # Objective: maximize sum_i probs[i] * p[i]
    #
    # Constraints:
    # 1. Feasibility: 0 <= x[i,j] <= 1 for all i,j
    # 2. Individual rationality: p[i] <= sum_j x[i,j] * values[i,j] for all i
    # 3. Incentive compatibility (truthfulness): For all i, i':
    #    p[i] - sum_j x[i,j]*values[i',j] >= p[i'] - sum_j x[i',j]*values[i',j]
    #    i.e., utility of truthfully reporting i >= utility of misreporting i'
    #    u(i, i) >= u(i, i') for all i, i'
    #    u(i, i') = sum_j x[i',j]*values[i,j] - p[i']
    #    So: sum_j x[i,j]*values[i,j] - p[i] >= sum_j x[i',j]*values[i,j] - p[i']
    #    Rearranged: -sum_j x[i,j]*values[i,j] + p[i] + sum_j x[i',j]*values[i,j] - p[i'] <= 0
    #    Or: sum_j (x[i',j] - x[i,j])*values[i,j] + p[i] - p[i'] <= 0
    
    # Let's use LP with scipy.optimize.linprog
    # Variable ordering: x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], ..., p[15]
    
    # Number of variables
    n_x = n * m  # 32
    n_p = n      # 16
    n_vars = n_x + n_p  # 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog form (minimize): c = [0, ..., 0, -probs[0], ..., -probs[15]]
    c = np.zeros(n_vars)
    c[n_x:] = -probs
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_ub = []
    
    # Individual rationality: p[i] - sum_j x[i,j]*values[i,j] <= 0
    # => sum_j (-values[i,j])*x[i,j] + p[i] <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        for j in range(m):
            row[i*m + j] = -values[i, j]
        row[n_x + i] = 1.0  # p[i]
        A_rows.append(row)
        b_ub.append(0.0)
    
    # Incentive compatibility: For all i, i':
    # sum_j (x[i',j] - x[i,j])*values[i,j] + p[i] - p[i'] <= 0
    # => sum_j values[i,j]*x[i',j] - sum_j values[i,j]*x[i,j] + p[i] - p[i'] <= 0
    for i in range(n):
        for ip in range(n):
            row = np.zeros(n_vars)
            for j in range(m):
                row[ip*m + j] = values[i, j]     # x[ip,j] * values[i,j]
                row[i*m + j] -= values[i, j]     # -x[i,j] * values[i,j]
            row[n_x + i] = 1.0     # p[i]
            row[n_x + ip] -= 1.0   # -p[ip]
            A_rows.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_ub)
    
    # Bounds: x[i,j] in [0,1], p[i] in [0, infinity)
    bounds = []
    for i in range(n):
        for j in range(m):
            bounds.append((0, 1))
    for i in range(n):
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback: simple allocation
        allocations = np.zeros((n, m))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    if not result.success:
        # Fallback: simple allocation
        allocations = np.zeros((n, m))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_flat = result.x[:n_x]
    p_flat = result.x[n_x:]
    
    allocations = x_flat.reshape(n, m)
    payments = p_flat
    
    # Clean up numerical noise
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify constraints
    # Check individual rationality
    for i in range(n):
        utility = np.sum(allocations[i] * values[i]) - payments[i]
        if utility < -1e-6:
            # Fix by adjusting payment
            payments[i] = np.sum(allocations[i] * values[i])
    
    # Check incentive compatibility
    for i in range(n):
        for ip in range(n):
            u_truth = np.sum(allocations[i] * values[i]) - payments[i]
            u_misrep = np.sum(allocations[ip] * values[i]) - payments[ip]
            if u_misrep - u_truth > 1e-6:
                # This shouldn't happen if LP solved correctly
                # Adjust payment to fix
                payments[ip] = np.sum(allocations[ip] * values[i]) - u_truth + 1e-6
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
