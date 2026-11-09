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
    DEADLINE = 80.0  # safety margin before 90s kill
    
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
        elif 'distribution' in distribution_spec:
            dist_name = str(distribution_spec['distribution'])
            probs = _infer_probs(dist_name, type_profiles, n_t)
        else:
            dist_name = str(distribution_spec)
            probs = _infer_probs(dist_name, type_profiles, n_t)
    else:
        dist_name = str(distribution_spec)
        probs = _infer_probs(dist_name, type_profiles, n_t)
    
    if probs is None:
        probs = np.ones(n_t) / n_t
    
    # LP formulation using the fact that for a single additive bidder,
    # the payment is determined by the allocation function via the
    # envelope theorem. We use a reduced LP formulation.
    
    from scipy.optimize import linprog
    
    # Variables: x[t, 0], x[t, 1] for t in range(n_t) -> 32 variables
    # The payment p[t] is implied by IC/IR constraints.
    # For a single bidder, the payment is:
    # p[t] = v1[t]*x[t,0] + v2[t]*x[t,1] - u[t]
    # where u[t] >= 0 is the utility of type t.
    # IC implies u is supermodular in a specific way.
    
    # Alternative: keep the original LP but use a more robust solver call.
    n_x = n_t * 2  # allocation variables
    n_p = n_t      # payment variables
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_t):
        c[n_x + t] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: v1[t]*x[t,0] + v2[t]*x[t,1] - p[t] >= 0
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1
        row[t * 2 + 1] = -v2
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints
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
    
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for t in range(n_t):
        bounds.append((0, 100))
    
    # Solve LP with tighter tolerances
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_staleness_tolerance': 1e-12,
                              'ipm_optimality_tolerance': 1e-12})
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        p_flat = result.x[n_x:n_x + n_p]
        
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
        # Fallback
        allocations = np.ones((n_t, 2))
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            payments[t] = (v1 + v2) / 2.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }

def _infer_probs(dist_name, type_profiles, n_t):
    """Infer probabilities from distribution name."""
    dist_lower = dist_name.lower()
    if 'uniform' in dist_lower or dist_lower in ('u', 'unif'):
        return np.ones(n_t) / n_t
    elif 'geometric' in dist_lower or 'geom' in dist_lower:
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (0.5 ** (v1 + v2))
        probs /= probs.sum()
        return probs
    elif 'beta' in dist_lower:
        return np.ones(n_t) / n_t
    else:
        return np.ones(n_t) / n_t
# EVOLVE-BLOCK-END
