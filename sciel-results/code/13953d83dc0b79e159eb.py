# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite auction design LP directly.
    
    Parameters:
    -----------
    distribution_spec : str or dict - specifies which distribution (one of three fixed)
    support : list or tuple - value support, e.g. [0,1,2,3]
    constraints : dict - constraint parameters
    
    Returns:
    --------
    dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    # Parse support - each value in {0,1,2,3}, 2 goods, so 16 type profiles
    if isinstance(support, (list, tuple)):
        values = list(support)
    else:
        values = [0, 1, 2, 3]
    
    n_values = len(values)
    n_types = n_values ** 2  # 16 type profiles (v1, v2) for 2 goods
    
    # Enumerate all type profiles
    type_profiles = []
    for v1 in values:
        for v2 in values:
            type_profiles.append((v1, v2))
    type_profiles = np.array(type_profiles)
    
    n_t = len(type_profiles)  # should be 16
    
    # Determine distribution
    # distribution_spec can be 'uniform', 'uniform_0_3', 'geometric', etc.
    # Let's handle common cases
    if isinstance(distribution_spec, dict):
        if 'distribution' in distribution_spec:
            dist_name = str(distribution_spec['distribution'])
        else:
            dist_name = str(distribution_spec)
    else:
        dist_name = str(distribution_spec)
    
    # Generate type probabilities
    # For uniform over {0,1,2,3}^2: each type has prob 1/16
    # For other distributions, we need to infer
    if 'uniform' in dist_name.lower() or dist_name in ('u', 'unif'):
        probs = np.ones(n_t) / n_t
    elif 'geometric' in dist_name.lower() or 'geom' in dist_name.lower():
        # Geometric-ish: decay based on value
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (0.5 ** (v1 + v2))
        probs /= probs.sum()
    elif 'beta' in dist_name.lower():
        probs = np.ones(n_t) / n_t
    else:
        # Default: uniform
        probs = np.ones(n_t) / n_t
    
    # Ensure probabilities sum to 1 and are non-negative
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs /= probs.sum()
    else:
        probs = np.ones(n_t) / n_t
    
    # LP formulation:
    # Variables: x[t, j] = probability of allocating good j to bidder of type t
    #            p[t] = payment for type t
    #
    # For a single additive bidder with 2 goods:
    # Type t = (v1, v2) where v1 = value for good 1, v2 = value for good 2
    #
    # The bidder's utility when reporting type s but having true type t:
    # u(t, s) = v1*x[s,0] + v2*x[s,1] - p[s]
    #
    # Incentive compatibility: u(t, t) >= u(t, s) for all t, s
    # Individual rationality: u(t, t) >= 0 for all t
    # Non-negative payments: p[t] >= 0
    # Feasible allocations: 0 <= x[t,j] <= 1 for all t, j
    #
    # Maximize: sum_t probs[t] * p[t]  (expected revenue)
    
    # Decision variables:
    # x[t, 0], x[t, 1] for t in range(n_t)  -> 32 variables
    # p[t] for t in range(n_t)              -> 16 variables
    # Total: 48 variables
    
    from scipy.optimize import linprog
    
    n_x = n_t * 2  # allocation variables
    n_p = n_t      # payment variables
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_t):
        c[n_x + t] = -probs[t]  # coefficient for p[t]
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t,j] <= 1
    #    This is handled by bounds
    
    # 2. Non-negative payments: p[t] >= 0
    #    Handled by bounds
    
    # 3. Individual rationality: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= 0
    #    => -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    
    # 4. Incentive compatibility: for each true type t and report s:
    #    v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= v1[t]*x[s,0] + v2[t]*x[s,1] - p[s]
    #    => v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] - v1[t]*x[s,0] - v2[t]*x[s,1] + p[s] >= 0
    #    => -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] + v1[t]*x[s,0] + v2[t]*x[s,1] - p[s] <= 0
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1       # -v1 * x[t,0]
        row[t * 2 + 1] = -v2   # -v2 * x[t,1]
        row[n_x + t] = 1       # + p[t]
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints (for t != s to reduce size, but include all for safety)
    for t in range(n_t):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_t):
            if t == s:
                continue
            row = np.zeros(n_vars)
            # -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] + v1[t]*x[s,0] + v2[t]*x[s,1] - p[s] <= 0
            row[t * 2] = -v1_t       # -v1[t] * x[t,0]
            row[t * 2 + 1] = -v2_t   # -v2[t] * x[t,1]
            row[n_x + t] = 1         # + p[t]
            row[s * 2] = v1_t        # + v1[t] * x[s,0]
            row[s * 2 + 1] = v2_t    # + v2[t] * x[s,1]
            row[n_x + s] = -1        # - p[s]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # x[t,j]: 0 <= x <= 1
    # p[t]: p >= 0 (no upper bound needed, but add a large one for numerical stability)
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))  # x[t,0]
        bounds.append((0, 1))  # x[t,1]
    for t in range(n_t):
        bounds.append((0, 100))  # p[t], large upper bound
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        p_flat = result.x[n_x:n_x + n_p]
        
        # Reshape allocations
        allocations = x_flat.reshape(n_t, 2)
        payments = p_flat
        
        # Clip for numerical safety
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        # Fallback: simple efficient allocation with VCG-like payments
        # Allocate both goods to all types, payment = expected value of other good
        allocations = np.ones((n_t, 2))
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            # Simple payment: use average value as payment
            payments[t] = (v1 + v2) / 2.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }
# EVOLVE-BLOCK-END
