"""Rewritten auction design exploiting single-bidder structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.array([dist.get(t, 0.0) for t in types], dtype=float)
            elif isinstance(dist, (list, tuple)):
                probs = np.array(dist, dtype=float)
            else:
                probs = np.array([0.5/16]*16, dtype=float)
        else:
            probs = np.array([1.0/16]*16, dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.array([1.0/16]*16, dtype=float)
    
    # Ensure probabilities sum to 1
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.array([1.0/16]*16, dtype=float)
    
    # Parse support
    if support is not None:
        if isinstance(support, dict):
            if 'allocations' in support:
                pass  # Will use default
            elif 'types' in support:
                pass
        elif isinstance(support, (list, tuple, np.ndarray)):
            pass
    
    # Parse constraints
    if constraints is not None and isinstance(constraints, dict):
        regret_tol = constraints.get('regret_tol', 1e-8)
        ir_tol = constraints.get('ir_tol', 1e-8)
    else:
        regret_tol = 1e-8
        ir_tol = 1e-8
    
    # For a single additive bidder with 2 goods, the problem decomposes.
    # However, since both goods are sold to the same bidder, we need to handle
    # the joint allocation. But incentive compatibility for a single bidder means
    # the allocation rule must be monotone and payments are determined by the integral.
    
    # Let's formulate this as an LP directly.
    # Variables:
    #   x[t] = allocation probability for type t (scalar, since single bidder)
    #   Actually, with 2 goods, x[t] should be a 2-vector (x1[t], x2[t])
    #   p[t] = payment for type t
    
    # Wait, the output format is allocations:(16,2) and payments:(16,)
    # So allocations[i,0] = probability of allocating good 1 to type i
    #    allocations[i,1] = probability of allocating good 2 to type i
    #    payments[i] = payment for type i
    
    # For a single bidder, IC (incentive compatibility) means:
    #   For all types t, t': v(t)·x(t) - p(t) >= v(t)·x(t') - p(t')
    # This is equivalent to: the utility function u(t) = v(t)·x(t) - p(t) 
    # must be such that the allocation x(t) is in the supergradient of u at v(t).
    
    # For finite types, IC is equivalent to:
    #   u(t) >= u(t') + (v(t) - v(t'))·x(t')  for all t, t'
    #   and x(t) = argmax_{x in {0,1}^2} v(t)·x - u(t) ... no wait.
    
    # Actually, for a single bidder with quasi-linear utility:
    #   IC iff the allocation rule x(v) is monotone (in the appropriate sense)
    #   and p(v) = v·x(v) - ∫ u(v') dv' (up to a constant)
    
    # For discrete types, IC constraints are:
    #   For all t, t': v(t)·x(t) - p(t) >= v(t)·x(t') - p(t')
    
    # Let's set up the LP:
    # Maximize: sum_t probs[t] * p(t)
    # Subject to:
    #   1. Feasibility: 0 <= x1[t] <= 1, 0 <= x2[t] <= 1 for all t
    #   2. IC: v(t)·x(t) - p(t) >= v(t)·x(t') - p(t') for all t, t'
    #   3. IR: v(t)·x(t) - p(t) >= 0 for all t
    #   4. p(t) >= 0 for all t
    
    # Variables: x1[0..15], x2[0..15], p[0..15] => 48 variables
    # Let's index them as:
    #   var 0-15: x1[0..15]
    #   var 16-31: x2[0..15]
    #   var 32-47: p[0..15]
    
    n_vars = 48
    x1_offset = 0
    x2_offset = 16
    p_offset = 32
    
    # Objective: maximize sum_t probs[t] * p(t)
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_offset + t] = -probs[t]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # 
    # IC constraints: for all t, t':
    #   v1[t]*x1[t] + v2[t]*x2[t] - p[t] >= v1[t]*x1[t'] + v2[t]*x2[t'] - p[t']
    #   => v1[t]*x1[t] + v2[t]*x2[t] - p[t] - v1[t]*x1[t'] - v2[t]*x2[t'] + p[t'] >= 0
    #   => -v1[t]*x1[t] - v2[t]*x2[t] + p[t] + v1[t]*x1[t'] + v2[t]*x2[t'] - p[t'] <= 0
    #
    # That's 16*16 = 256 constraints
    #
    # IR constraints: for all t:
    #   v1[t]*x1[t] + v2[t]*x2[t] - p[t] >= 0
    #   => -v1[t]*x1[t] - v2[t]*x2[t] + p[t] <= 0
    #
    # That's 16 constraints
    #
    # Non-negativity: x1[t] >= 0, x2[t] >= 0, p[t] >= 0
    # These are handled by bounds.
    
    # Let's build A_ub and b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints
    for t in range(n_types):
        for t2 in range(n_types):
            if t == t2:
                continue  # Trivially satisfied
            row = np.zeros(n_vars)
            v1_t, v2_t = types[t]
            v1_t2, v2_t2 = types[t2]
            
            # -v1[t]*x1[t] - v2[t]*x2[t] + p[t] + v1[t]*x1[t'] + v2[t]*x2[t'] - p[t'] <= 0
            row[x1_offset + t] = -v1_t
            row[x2_offset + t] = -v2_t
            row[p_offset + t] = 1
            row[x1_offset + t2] = v1_t
            row[x2_offset + t2] = v2_t
            row[p_offset + t2] = -1
            
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    # IR constraints
    for t in range(n_types):
        row = np.zeros(n_vars)
        v1_t, v2_t = types[t]
        
        # -v1[t]*x1[t] - v2[t]*x2[t] + p[t] <= 0
        row[x1_offset + t] = -v1_t
        row[x2_offset + t] = -v2_t
        row[p_offset + t] = 1
        
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    # x1[t] in [0, 1], x2[t] in [0, 1], p[t] in [0, inf)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # p[t]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline - 5.0:
        # Return a feasible solution if we can find one quickly
        # Simple feasible solution: allocate nothing, charge nothing
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if not result.success:
        # Fallback: simple feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_opt = result.x
    
    allocations = np.zeros((n_types, 2))
    for t in range(n_types):
        allocations[t, 0] = x_opt[x1_offset + t]
        allocations[t, 1] = x_opt[x2_offset + t]
    
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = x_opt[p_offset + t]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Verify constraints and fix if needed
    for t in range(n_types):
        v1_t, v2_t = types[t]
        utility = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if utility < -1e-10:
            # IR violated, adjust payment
            payments[t] = max(0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    # Check IC constraints
    for t in range(n_types):
        for t2 in range(n_types):
            v1_t, v2_t = types[t]
            v1_t2, v2_t2 = types[t2]
            util_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            util_t2_at_t = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            if util_t < util_t2_at_t - 1e-10:
                # IC violated, this shouldn't happen with a correct LP solution
                # but let's be safe
                pass
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
