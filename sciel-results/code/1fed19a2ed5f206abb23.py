"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

# Cache for baseline LP solutions per distribution spec
_lp_cache = {}

def _solve_baseline_lp(dist_spec, support, constraints):
    """Independently solve the finite-support mechanism LP to get the optimal
    expected revenue under feasibility, IR, and incentive compatibility.
    This serves as the normalization denominator."""
    if dist_spec in _lp_cache:
        return _lp_cache[dist_spec]
    
    # Parse support: list of (value1, value2) or similar
    # Support is typically a list of tuples representing possible value profiles
    # Each bidder has one type, so support is a list of types (v1, v2)
    types = support
    n_types = len(types)
    
    # Probabilities from distribution_spec
    # distribution_spec is a dict or list of probabilities
    if isinstance(dist_spec, dict):
        # Assume keys are type indices or tuples
        probs = np.array([dist_spec.get(i, 0.0) for i in range(n_types)], dtype=float)
    elif isinstance(dist_spec, (list, tuple)):
        probs = np.array(dist_spec, dtype=float)
        if len(probs) != n_types:
            # Try to match by value
            probs = np.zeros(n_types)
            for i, t in enumerate(types):
                if i < len(dist_spec):
                    probs[i] = dist_spec[i]
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: allocation x[t, g] for type t, good g, and payment p[t] for type t
    # n_types * 2 + n_types variables
    n_vars = n_types * 3  # x1, x2, p for each type
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog, we minimize, so c = -probs[t] for payment variables
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[t * 3 + 2] = -probs[t]  # payment variable index
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t,g] <= 1
    # 2. Individual Rationality: x[t,1]*v[t,1] + x[t,2]*v[t,2] - p[t] >= 0
    #    => -v[t,1]*x[t,1] - v[t,2]*x[t,2] + p[t] <= 0
    # 3. Incentive Compatibility (Truthfulness):
    #    For each type t and report t':
    #    x[t',1]*v[t,1] + x[t',2]*v[t,2] - p[t'] <= x[t,1]*v[t,1] + x[t,2]*v[t,2] - p[t]
    #    => (x[t',1]-x[t,1])*v[t,1] + (x[t',2]-x[t,2])*v[t,2] - (p[t']-p[t]) <= 0
    
    A_ub = []
    b_ub = []
    
    # IR constraints
    for t in range(n_types):
        v1, v2 = types[t]
        row = np.zeros(n_vars)
        row[t * 3 + 0] = -v1
        row[t * 3 + 1] = -v2
        row[t * 3 + 2] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # IC constraints
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            v1_t, v2_t = types[t]
            row = np.zeros(n_vars)
            row[t_prime * 3 + 0] = v1_t
            row[t_prime * 3 + 1] = v2_t
            row[t_prime * 3 + 2] = -1
            row[t * 3 + 0] = -v1_t
            row[t * 3 + 1] = -v2_t
            row[t * 3 + 2] = 1
            A_ub.append(row)
            b_ub.append(0)
    
    # Bounds: x in [0,1], p in [0, inf)
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1
        bounds.append((0, 1))  # x2
        bounds.append((0, None))  # p
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        optimal_revenue = -result.fun
    else:
        optimal_revenue = 1.0  # Fallback
    
    _lp_cache[dist_spec] = optimal_revenue
    return optimal_revenue


def _construct_auction_from_lp(dist_spec, support, constraints):
    """Solve the LP to find the optimal mechanism and return it."""
    types = support
    n_types = len(types)
    
    # Probabilities
    if isinstance(dist_spec, dict):
        probs = np.array([dist_spec.get(i, 0.0) for i in range(n_types)], dtype=float)
    elif isinstance(dist_spec, (list, tuple)):
        probs = np.array(dist_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    if probs.sum() > 0:
        probs = probs / probs.sum()
    
    n_vars = n_types * 3
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[t * 3 + 2] = -probs[t]
    
    A_ub = []
    b_ub = []
    
    for t in range(n_types):
        v1, v2 = types[t]
        row = np.zeros(n_vars)
        row[t * 3 + 0] = -v1
        row[t * 3 + 1] = -v2
        row[t * 3 + 2] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            v1_t, v2_t = types[t]
            row = np.zeros(n_vars)
            row[t_prime * 3 + 0] = v1_t
            row[t_prime * 3 + 1] = v2_t
            row[t_prime * 3 + 2] = -1
            row[t * 3 + 0] = -v1_t
            row[t * 3 + 1] = -v2_t
            row[t * 3 + 2] = 1
            A_ub.append(row)
            b_ub.append(0)
    
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, None))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple deterministic mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            v1, v2 = types[t]
            if v1 >= v2:
                allocations[t, 0] = 1
                payments[t] = v1
            else:
                allocations[t, 1] = 1
                payments[t] = v2
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for t in range(n_types):
        allocations[t, 0] = x[t * 3 + 0]
        allocations[t, 1] = x[t * 3 + 1]
        payments[t] = x[t * 3 + 2]
    
    # Clamp to valid range
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {"allocations": allocations, "payments": payments}


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: List of possible value profiles (v1, v2)
        constraints: Additional constraints (unused in current implementation)
    
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    try:
        # Solve the LP to find the optimal mechanism
        result = _construct_auction_from_lp(distribution_spec, support, constraints)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, return what we have
            pass
        
        return result
    
    except Exception as e:
        # Fallback: simple deterministic mechanism
        n_types = len(support)
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            v1, v2 = support[t]
            if v1 >= v2:
                allocations[t, 0] = 1
                payments[t] = v1
            else:
                allocations[t, 1] = 1
                payments[t] = v2
        return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
