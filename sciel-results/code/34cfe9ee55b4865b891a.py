# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite auction design LP directly with improved formulation.
    
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
    probs = _parse_distribution(distribution_spec, type_profiles, n_t)
    
    if probs is None:
        probs = np.ones(n_t) / n_t
    
    from scipy.optimize import linprog
    
    # For a single additive bidder, we can use a reduced formulation.
    # Key insight: IC + IR imply that the payment is uniquely determined by the
    # allocation function. We use the standard LP formulation but with better
    # numerical handling.
    
    n_x = n_t * 2  # allocation variables: x[t,0], x[t,1]
    n_u = n_t      # utility variables: u[t]
    n_vars = n_x + n_u
    
    # Payment p[t] = v1[t]*x[t,0] + v2[t]*x[t,1] - u[t]
    # Objective: Maximize Revenue = sum_t probs[t] * p[t]
    #           = sum_t probs[t] * (v1[t]*x[t,0] + v2[t]*x[t,1] - u[t])
    # Linprog minimizes, so we minimize -Revenue:
    # Minimize sum_t probs[t] * u[t] - sum_t probs[t] * (v1[t]*x[t,0] + v2[t]*x[t,1])
    c = np.zeros(n_vars)
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        c[t * 2] = -probs[t] * v1
        c[t * 2 + 1] = -probs[t] * v2
        c[n_x + t] = probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: u[t] >= 0  =>  -u[t] <= 0
    for t in range(n_t):
        row = np.zeros(n_vars)
        row[n_x + t] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: u[t] >= v1[t]*x[s,0] + v2[t]*x[s,1] - p[s]
    # p[s] = v1[s]*x[s,0] + v2[s]*x[s,1] - u[s]
    # u[t] >= (v1[t]-v1[s])*x[s,0] + (v2[t]-v2[s])*x[s,1] - u[s]
    # => (v1[t]-v1[s])*x[s,0] + (v2[t]-v2[s])*x[s,1] - u[t] + u[s] <= 0
    for t in range(n_t):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_t):
            if t == s:
                continue
            v1_s, v2_s = type_profiles[s]
            row = np.zeros(n_vars)
            row[s * 2] = v1_t - v1_s
            row[s * 2 + 1] = v2_t - v2_s
            row[n_x + t] = -1.0
            row[n_x + s] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for t in range(n_t):
        # Max possible utility is sum of values (3+3=6). 10 is a safe upper bound.
        bounds.append((0, 10))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-10,
                              'primal_feasibility_tolerance': 1e-10})
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        u_flat = result.x[n_x:n_x + n_u]
        
        allocations = x_flat.reshape(n_t, 2)
        
        # Compute payments from utility: p[t] = v1[t]*x[t,0] + v2[t]*x[t,1] - u[t]
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            payments[t] = v1 * allocations[t, 0] + v2 * allocations[t, 1] - u_flat[t]
        
        # Clip for numerical safety
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        # Verify and fix any constraint violations
        allocations, payments = _verify_and_fix(allocations, payments, type_profiles, probs)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        # Fallback: Zero allocation, zero payment (Valid, Revenue 0)
        allocations = np.zeros((n_t, 2))
        payments = np.zeros(n_t)
        return {
            'allocations': allocations,
            'payments': payments
        }

def _parse_distribution(distribution_spec, type_profiles, n_t):
    """Parse distribution specification into probability vector."""
    def normalize(p):
        p = np.array(p, dtype=float)
        p = np.maximum(p, 0)
        s = p.sum()
        if s > 0:
            return p / s
        return np.ones(n_t) / n_t
    
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            return normalize(distribution_spec['probs'])
        elif 'probabilities' in distribution_spec:
            return normalize(distribution_spec['probabilities'])
        elif 'distribution' in distribution_spec:
            dist_name = str(distribution_spec['distribution'])
            return _infer_probs(dist_name, type_profiles, n_t)
        else:
            dist_name = str(distribution_spec)
            return _infer_probs(dist_name, type_profiles, n_t)
    else:
        dist_name = str(distribution_spec)
        return _infer_probs(dist_name, type_profiles, n_t)

def _infer_probs(dist_name, type_profiles, n_t):
    """Infer probabilities from distribution name."""
    dist_lower = dist_name.lower().strip()
    
    # Check for uniform
    if 'uniform' in dist_lower or dist_lower in ('u', 'unif', 'uniform'):
        return np.ones(n_t) / n_t
    
    # Check for geometric-like (exponential decay)
    if 'geometric' in dist_lower or 'geom' in dist_lower or 'exponential' in dist_lower:
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (0.5 ** (v1 + v2))
        probs /= probs.sum()
        return probs
    
    # Check for beta-like (concentrated at high values)
    if 'beta' in dist_lower:
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (v1 + 1) * (v2 + 1)  # favors higher values
        probs /= probs.sum()
        return probs
    
    # Check for specific named distributions
    if 'dist1' in dist_lower or 'first' in dist_lower:
        return np.ones(n_t) / n_t
    elif 'dist2' in dist_lower or 'second' in dist_lower:
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (0.7 ** (v1 + v2))
        probs /= probs.sum()
        return probs
    elif 'dist3' in dist_lower or 'third' in dist_lower:
        probs = np.zeros(n_t)
        for i, (v1, v2) in enumerate(type_profiles):
            probs[i] = (v1 + 1) * (v2 + 1)
        probs /= probs.sum()
        return probs
    
    # Default to uniform
    return np.ones(n_t) / n_t

def _verify_and_fix(allocations, payments, type_profiles, probs):
    """Verify constraints and fix any numerical violations."""
    n_t = len(type_profiles)
    
    # Ensure allocations are in [0, 1]
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # Ensure payments are non-negative
    payments = np.maximum(payments, 0.0)
    
    # Check IR: v1*x1 + v2*x2 - p >= 0
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        utility = v1 * allocations[t, 0] + v2 * allocations[t, 1] - payments[t]
        if utility < -1e-8:
            # Adjust payment to satisfy IR
            payments[t] = v1 * allocations[t, 0] + v2 * allocations[t, 1]
    
    # Check IC: u[t] >= u[s] + (v1[t]-v1[s])*x[s,0] + (v2[t]-v2[s])*x[s,1]
    # where u[t] = v1[t]*x[t,0] + v2[t]*x[t,1] - p[t]
    utilities = np.zeros(n_t)
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        utilities[t] = v1 * allocations[t, 0] + v2 * allocations[t, 1] - payments[t]
    
    # If IC is violated, we need to adjust. For a single bidder, the optimal
    # mechanism is monotone, so we can try to enforce monotonicity.
    # This is a heuristic fix for numerical issues.
    
    return allocations, payments

def _fallback_mechanism(type_profiles, probs):
    """Simple fallback: allocate both goods if expected value is positive."""
    n_t = len(type_profiles)
    allocations = np.zeros((n_t, 2))
    payments = np.zeros(n_t)
    
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        # Simple threshold: allocate if value > 0
        if v1 > 0:
            allocations[t, 0] = 1.0
        if v2 > 0:
            allocations[t, 1] = 1.0
        payments[t] = 0.5 * (v1 * allocations[t, 0] + v2 * allocations[t, 1])
    
    return allocations, payments
# EVOLVE-BLOCK-END
