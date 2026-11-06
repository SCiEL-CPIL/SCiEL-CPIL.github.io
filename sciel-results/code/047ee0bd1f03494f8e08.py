"""Incentive-compatible finite auction design via direct LP with utility variables."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction maximizing expected revenue.
    
    Args:
        distribution_spec: Dict with probability distribution over types.
        support: List of (v1, v2) tuples, the 16 type values.
        constraints: Dict with additional constraints.
    
    Returns:
        dict: {'allocations': np.array shape (16,2), 'payments': np.array shape (16,)}
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse the 16 types: values in {0,1,2,3} for each of 2 goods
    # Types indexed 0..15, type i has values (i//4, i%4) or similar
    # Let's figure out the type-value mapping from support
    n_types = 16
    n_goods = 2
    
    # Build type-value matrix: V[i] = (v1_i, v2_i) for type i
    # support should list the 16 value vectors
    if support is not None and len(support) == 16:
        V = np.array(support, dtype=float)
    else:
        # Default: all combinations of {0,1,2,3}^2
        V = np.array([[a, b] for a in range(4) for b in range(4)], dtype=float)
    
    # Get probabilities
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float)
            elif 'prob' in distribution_spec:
                probs = np.array(distribution_spec['prob'], dtype=float)
            elif 'distribution' in distribution_spec:
                probs = np.array(distribution_spec['distribution'], dtype=float)
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    probs = probs / probs.sum()  # normalize
    
    # LP formulation:
    # Variables: x[0..31] = allocations (16 types * 2 goods), u[0..15] = utilities
    # Total vars: 48
    #
    # Objective: maximize E[sum_g v_ig * x_ig - u_i] = sum_i p_i * (V[i]·x[i] - u[i])
    # = sum_i p_i * (sum_g V[i,g] * x[i,g]) - sum_i p_i * u_i
    #
    # For linprog (minimization): minimize -objective
    # c vector: for x[i,g]: -p_i * V[i,g], for u[i]: p_i
    #
    # Constraints:
    # 1. IC: for all i, j: u_i - u_j >= V[i]·x[j] - V[j]·x[j]  ... wait
    #    IC: u_i >= u_j + (V[i] - V[j])·x_j  for all i,j
    #    Rearranged: u_i - u_j - (V[i]-V[j])·x_j >= 0
    #    Or: -(u_i - u_j - (V[i]-V[j])·x_j) <= 0
    #    
    #    Actually, the standard IC condition for quasi-linear utility:
    #    u_i = V[i]·x[i] - p_i
    #    IC: u_i >= V[i]·x[j] - p_j for all j  (truthful reporting is optimal)
    #    => V[i]·x[i] - p_i >= V[i]·x[j] - p_j
    #    => (V[i]·x[i] - V[i]·x[j]) >= (p_i - p_j)
    #    => (V[i]·x[i] - V[i]·x[j]) + u_j - u_i >= 0  [since u = V·x - p]
    #    
    #    Let me use utility variables directly.
    #    u_i = V[i]·x[i] - p_i
    #    p_i = V[i]·x[i] - u_i
    #    
    #    IC: for all i, j: u_i >= V[i]·x[j] - p_j = V[i]·x[j] - V[j]·x[j] + u_j
    #    => u_i - u_j >= V[i]·x[j] - V[j]·x[j]
    #    => u_i - u_j - V[i]·x[j] + V[j]·x[j] >= 0
    #    => -(u_i - u_j - V[i]·x[j] + V[j]·x[j]) <= 0
    #    
    # 2. IR: u_i >= 0 for all i
    # 3. Feasibility: 0 <= x[i,g] <= 1 for all i,g
    
    n_vars = 2 * n_types + n_types  # 32 allocation + 16 utility = 48
    # x indices: 0..31, u indices: 32..47
    
    def x_idx(i, g):
        return i * n_goods + g
    
    def u_idx(i):
        return 2 * n_types + i
    
    # Objective: minimize -E[revenue] = -sum_i p_i * (V[i]·x[i] - u_i)
    c = np.zeros(n_vars)
    for i in range(n_types):
        for g in range(n_goods):
            c[x_idx(i, g)] = -probs[i] * V[i, g]
        c[u_idx(i)] = probs[i]  # because we're minimizing, and objective has -u_i
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: for all i, j: u_i - u_j - V[i]·x[j] + V[j]·x[j] >= 0
    # => -(u_i - u_j - V[i]·x[j] + V[j]·x[j]) <= 0
    # Row: -u_i + u_j + V[i]·x[j] - V[j]·x[j] <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[u_idx(i)] = -1.0
            row[u_idx(j)] = 1.0
            for g in range(n_goods):
                row[x_idx(j, g)] = V[i, g] - V[j, g]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR: u_i >= 0 => -u_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(i)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Feasibility: x[i,g] <= 1 => x[i,g] <= 1
    for i in range(n_types):
        for g in range(n_goods):
            row = np.zeros(n_vars)
            row[x_idx(i, g)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # x[i,g] >= 0 => -x[i,g] <= 0
    for i in range(n_types):
        for g in range(n_goods):
            row = np.zeros(n_vars)
            row[x_idx(i, g)] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds for linprog
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))  # u_i >= 0, unbounded above
    
    # Solve with HiGHS for accuracy
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
        options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                 'primal_feasibility_tolerance': 1e-10,
                 'ipm_optimality_tolerance': 1e-10}
    )
    
    if not result.success:
        # Fallback: return zero allocations and zero payments
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x_sol = result.x
    
    allocations = np.zeros((16, 2))
    utilities = np.zeros(16)
    for i in range(n_types):
        for g in range(n_goods):
            allocations[i, g] = x_sol[x_idx(i, g)]
        utilities[i] = x_sol[u_idx(i)]
    
    # Compute payments: p_i = V[i]·x[i] - u_i
    payments = np.zeros(16)
    for i in range(n_types):
        payments[i] = np.dot(V[i], allocations[i]) - utilities[i]
    
    # Clamp small negative values to zero (numerical noise)
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify IC and IR, fix if needed
    # IC: u_i >= V[i]·x[j] - p_j for all i,j
    # u_i = V[i]·x[i] - p_i
    # Check: V[i]·x[i] - p_i >= V[i]·x[j] - p_j
    for i in range(n_types):
        for j in range(n_types):
            lhs = np.dot(V[i], allocations[i]) - payments[i]
            rhs = np.dot(V[i], allocations[j]) - payments[j]
            if lhs < rhs - 1e-8:
                # Fix: adjust p_i down to satisfy IC
                # V[i]·x[i] - p_i >= V[i]·x[j] - p_j
                # p_i <= V[i]·x[i] - V[i]·x[j] + p_j
                max_p_i = np.dot(V[i], allocations[i]) - np.dot(V[i], allocations[j]) + payments[j]
                if payments[i] > max_p_i:
                    payments[i] = max(0.0, max_p_i)
    
    # Re-verify IR
    for i in range(n_types):
        util_i = np.dot(V[i], allocations[i]) - payments[i]
        if util_i < -1e-8:
            payments[i] = max(0.0, np.dot(V[i], allocations[i]))
    
    # Final clamp
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
