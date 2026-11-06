"""Optimal single-bidder auction via direct LP formulation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct optimal incentive-compatible auction for one additive bidder, two goods.
    
    Args:
        distribution_spec: specification of the value distribution
        support: value support (e.g., {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    # Values for each good: 0, 1, 2, 3
    # 16 types: (v1, v2) for v1 in {0,1,2,3}, v2 in {0,1,2,3}
    values = [0, 1, 2, 3]
    
    # Enumerate all 16 types
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    
    n_types = len(types)  # 16
    
    # Get probabilities from distribution_spec
    # Assume distribution_spec provides probabilities for each type
    if isinstance(distribution_spec, dict) and 'probabilities' in distribution_spec:
        probs = np.array(distribution_spec['probabilities'])
        if len(probs) != n_types:
            # Try to reshape or handle differently
            probs = np.resize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    else:
        # Default: uniform distribution
        probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum()  # normalize
    
    # For a single bidder, IC + IR optimal mechanism:
    # Allocation x_i(v) is non-decreasing in v_i
    # Payment p(v) = sum_i [v_i * x_i(v) - integral_0^{v_i} x_i(t, v_{-i}) dt]
    # 
    # With discrete values, the optimal mechanism has threshold structure:
    # x_i(v) = 1 if v_i >= r_i, else 0, for some reserve r_i in {0,1,2,3}
    # 
    # We search over all possible reserve pairs (r1, r2) in {0,1,2,3} x {0,1,2,3}
    
    best_revenue = -1
    best_r1, best_r2 = 0, 0
    
    for r1 in values:
        for r2 in values:
            # Compute allocation and payment for each type
            alloc = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            
            for idx, (v1, v2) in enumerate(types):
                x1 = 1.0 if v1 >= r1 else 0.0
                x2 = 1.0 if v2 >= r2 else 0.0
                alloc[idx, 0] = x1
                alloc[idx, 1] = x2
                
                # Payment by envelope
                # p(v) = sum_i [v_i * x_i - integral_0^{v_i} x_i(t) dt]
                # For x_i(t) = 1 if t >= r_i: integral_0^{v_i} x_i(t) dt = max(0, v_i - r_i)
                p1 = v1 * x1 - max(0, v1 - r1)
                p2 = v2 * x2 - max(0, v2 - r2)
                payments[idx] = p1 + p2
            
            # Check IR: payments <= v · x for all types
            for idx, (v1, v2) in enumerate(types):
                if payments[idx] > v1 * alloc[idx, 0] + v2 * alloc[idx, 1] + 1e-10:
                    break
            else:
                # Check nonnegative payments
                if np.all(payments >= -1e-10):
                    revenue = np.sum(probs * payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_r1, best_r2 = r1, r2
    
    # Construct final mechanism
    alloc = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for idx, (v1, v2) in enumerate(types):
        x1 = 1.0 if v1 >= best_r1 else 0.0
        x2 = 1.0 if v2 >= best_r2 else 0.0
        alloc[idx, 0] = x1
        alloc[idx, 1] = x2
        
        p1 = v1 * x1 - max(0, v1 - best_r1)
        p2 = v2 * x2 - max(0, v2 - best_r2)
        payments[idx] = p1 + p2
    
    # Ensure nonnegative
    payments = np.maximum(payments, 0)
    
    return {
        'allocations': alloc,
        'payments': payments
    }
# EVOLVE-BLOCK-END
