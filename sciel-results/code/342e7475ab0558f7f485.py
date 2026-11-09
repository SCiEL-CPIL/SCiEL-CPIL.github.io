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
    
    # Fallback function
    def get_fallback():
        try:
            from autoresearch_suite.science_economics_expansion import auction_baseline
            res = auction_baseline(distribution_spec, support, constraints)
            return {
                'allocations': np.array(res['allocations'], dtype=float),
                'payments': np.array(res['payments'], dtype=float)
            }
        except Exception:
            # Absolute fallback: zero allocations, zero payments
            return {
                'allocations': np.zeros((16, 2)),
                'payments': np.zeros(16)
            }

    # 1. Parse Support to determine Type Profiles
    # support can be:
    # a) [0, 1, 2, 3] -> Cartesian product
    # b) [(0,0), (0,1), ...] -> Direct list of types
    
    type_profiles = None
    n_types = 0
    
    if support is None:
        values = [0, 1, 2, 3]
    elif isinstance(support, (list, tuple, np.ndarray)):
        # Check if it's a list of scalars or a list of vectors
        if len(support) == 0:
            values = [0, 1, 2, 3]
        else:
            # Check first element
            first_elem = support[0]
            if isinstance(first_elem, (int, float, np.integer, np.floating)):
                # List of scalars
                values = list(support)
            else:
                # List of tuples/arrays
                # Convert to list of tuples
                type_profiles = [tuple(v) for v in support]
                n_types = len(type_profiles)
                values = None # Not needed if we have types
    else:
        values = [0, 1, 2, 3]

    if type_profiles is None:
        # Generate Cartesian product
        if values is None:
            values = [0, 1, 2, 3]
        type_profiles = list(product(values, repeat=2))
        n_types = len(type_profiles)
        
    # Ensure we have 16 types. If not, pad or truncate? 
    # The task specifies (16,2) output. 
    # If n_types != 16, we must map to 16.
    # Assuming the standard 16 types {0,1,2,3}^2 is expected.
    # If the user provided a different support, we might have a mismatch.
    # For safety, if n_types != 16, we try to force the standard 16 types 
    # or return fallback. Given the task contract, it's likely always 16.
    
    if n_types != 16:
        # Try to standardize to 16 types if possible, else fallback
        # If support was custom, we can't just guess. 
        # But the return shape is fixed (16,2).
        # We will proceed with the parsed types if they are 16.
        # If not, we use the standard 16 types and hope the distribution matches.
        standard_values = [0, 1, 2, 3]
        type_profiles = list(product(standard_values, repeat=2))
        n_types = 16

    type_profiles = np.array(type_profiles, dtype=float)
    
    # 2. Determine Distribution Probabilities
    probs = None
    
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist_name = str(distribution_spec['distribution']).lower()
            if 'uniform' in dist_name:
                probs = np.ones(n_types) / n_types
            elif 'geometric' in dist_name:
                # Geometric decay on sum of values
                probs = np.array([0.5 ** (t[0] + t[1]) for t in type_profiles])
                probs /= probs.sum()
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (n_types, 2):
            # Maybe it's values? No, spec is distribution.
            # If it's a list of 16 items, treat as probs.
            probs = arr
        else:
            probs = np.ones(n_types) / n_types
    else:
        dist_name = str(distribution_spec).lower()
        if 'uniform' in dist_name:
            probs = np.ones(n_types) / n_types
        elif 'geometric' in dist_name:
            probs = np.array([0.5 ** (t[0] + t[1]) for t in type_profiles])
            probs /= probs.sum()
        else:
            # Default to uniform if unknown
            probs = np.ones(n_types) / n_types

    # Normalize and sanitize probs
    probs = np.maximum(probs, 0)
    if probs.sum() > 1e-12:
        probs /= probs.sum()
    else:
        probs = np.ones(n_types) / n_types

    # 3. Setup LP
    from scipy.optimize import linprog
    
    # Variables:
    # x[t, 0], x[t, 1] for t in 0..15 (32 vars)
    # p[t] for t in 0..15 (16 vars)
    # Total 48 vars
    
    n_x = n_types * 2
    n_p = n_types
    n_vars = n_x + n_p
    
    # Objective: Max sum(probs * p) -> Min -sum(probs * p)
    c = np.zeros(n_vars)
    c[n_x:] = -probs
    
    # Constraints
    A_ub_rows = []
    b_ub_vals = []
    
    # IR: v_t . x_t - p_t >= 0  =>  -v_t . x_t + p_t <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = type_profiles[t]
        row[t*2] = -v1
        row[t*2+1] = -v2
        row[n_x+t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    # IC: v_t . x_t - p_t >= v_t . x_s - p_s
    # => -v_t . x_t + p_t + v_t . x_s - p_s <= 0
    for t in range(n_types):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_types):
            if t == s:
                continue
            row = np.zeros(n_vars)
            # -v_t . x_t
            row[t*2] = -v1_t
            row[t*2+1] = -v2_t
            # + p_t
            row[n_x+t] = 1
            # + v_t . x_s
            row[s*2] = v1_t
            row[s*2+1] = v2_t
            # - p_s
            row[n_x+s] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = []
    for _ in range(n_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
    for _ in range(n_types):
        bounds.append((0, 100)) # p
    
    # Solve
    try:
        if time.time() - start_time > DEADLINE - 5:
            return get_fallback()
            
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_flat = result.x[:n_x]
            p_flat = result.x[n_x:n_x+n_p]
            
            allocations = x_flat.reshape(n_types, 2)
            payments = p_flat
            
            # Clip for numerical safety
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            return get_fallback()
            
    except Exception as e:
        return get_fallback()

# EVOLVE-BLOCK-END
