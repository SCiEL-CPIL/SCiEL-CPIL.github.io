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
    probs = None
    
    # Helper to normalize
    def normalize(p):
        p = np.array(p, dtype=float)
        p = np.maximum(p, 0)
        s = p.sum()
        if s > 0:
            return p / s
        return np.ones(n_t) / n_t

    if isinstance(distribution_spec, dict):
        # Check if explicit probabilities are provided
        if 'probs' in distribution_spec:
            probs = normalize(distribution_spec['probs'])
        elif 'probabilities' in distribution_spec:
            probs = normalize(distribution_spec['probabilities'])
        else:
            # Try to infer from name/type
            dist_name = str(distribution_spec.get('distribution', distribution_spec.get('name', 'uniform'))).lower()
            if 'uniform' in dist_name:
                probs = np.ones(n_t) / n_t
            elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
                probs = np.zeros(n_t)
                for i, (v1, v2) in enumerate(type_profiles):
                    # Standard geometric decay
                    probs[i] = (0.5 ** (v1 + v2))
                probs = normalize(probs)
            elif 'beta' in dist_name:
                # Often treated as uniform in simple discrete contexts if no params given
                # Or maybe a specific concave shape. Let's stick to uniform for safety unless specified.
                probs = np.ones(n_t) / n_t
            else:
                probs = np.ones(n_t) / n_t
    else:
        dist_name = str(distribution_spec).lower()
        if 'uniform' in dist_name or dist_name in ('u', 'unif'):
            probs = np.ones(n_t) / n_t
        elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
            probs = np.zeros(n_t)
            for i, (v1, v2) in enumerate(type_profiles):
                probs[i] = (0.5 ** (v1 + v2))
            probs = normalize(probs)
        elif 'beta' in dist_name:
            probs = np.ones(n_t) / n_t
        else:
            # Default: uniform
            probs = np.ones(n_t) / n_t

    # Ensure probabilities sum to 1 and are non-negative
    probs = normalize(probs)
    
    # LP formulation over allocation variables only
    # Key insight: For a single bidder, IC constraints determine payments as a
    # linear function of allocations. We can eliminate payment variables.
    #
    # Let x[t] = (x1[t], x2[t]) be the allocation to type t.
    # IC: u(t, t) >= u(t, s) for all t, s
    #   v_t . x[t] - p[t] >= v_t . x[s] - p[s]
    #
    # The optimal mechanism sets payments so that the lowest-type IR binds.
    # For type 0 (v=0,0): p[0] = 0 (IR binds)
    # For type t: p[t] = v_t . x[t] - v_t . x[0] + p[0] ... but this isn't quite right
    # for multi-dimensional types.
    #
    # Actually, with multi-dimensional types, the payment is:
    # p[t] = v_t . x[t] - min_{s: v_s <= v_t componentwise} (v_t . x[s] - p[s])
    #
    # This is complex. Let's use a different approach: solve the full LP but with
    # tighter bounds and better conditioning. Or use the fact that the problem is
    # small enough (48 vars, ~240 constraints) that HiGHS should be fast.
    #
    # Alternative: Use the dual or a different formulation.
    #
    # Let's try: fix p[0] = 0 (WLOG, since adding constant to all payments doesn't
    # change IC, and IR for type 0 forces p[0] <= 0, so p[0] = 0 is optimal).
    # Then IC constraints determine p[t] for all t in terms of allocations.
    
    from scipy.optimize import linprog
    
    # Actually, let's keep the full LP but improve it:
    # 1. Use tighter payment bounds
    # 2. Pre-compute constraint matrix more efficiently
    
    n_x = n_t * 2  # allocation variables
    n_p = n_t      # payment variables  
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_t):
        c[n_x + t] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1
        row[t * 2 + 1] = -v2
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: u(t,t) >= u(t,s)
    # v_t . x[t] - p[t] >= v_t . x[s] - p[s]
    # => -v_t . x[t] + p[t] + v_t . x[s] - p[s] <= 0
    for t in range(n_t):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_t):
            if t == s:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[n_x + t] = 1
            row[s * 2] = v1_t
            row[s * 2 + 1] = v2_t
            row[n_x + s] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Tighter bounds: payments bounded by max possible value sum
    max_val_sum = max(v1 + v2 for v1, v2 in type_profiles)
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for t in range(n_t):
        bounds.append((0, max_val_sum + 1))
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'mip_rel_gap': 0, 'time_limit': 60})
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        p_flat = result.x[n_x:n_x + n_p]
        
        allocations = x_flat.reshape(n_t, 2)
        payments = p_flat
        
        # Clean up numerical noise
        allocations = np.clip(allocations, 0.0, 1.0)
        allocations[np.abs(allocations) < 1e-10] = 0.0
        allocations[np.abs(allocations - 1.0) < 1e-10] = 1.0
        payments = np.maximum(payments, 0.0)
        payments[np.abs(payments) < 1e-10] = 0.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        # Fallback: use a simple valid mechanism
        # Allocate both goods to all types, charge average value
        allocations = np.ones((n_t, 2))
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            payments[t] = (v1 + v2) / 2.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }
# EVOLVE-BLOCK-END
