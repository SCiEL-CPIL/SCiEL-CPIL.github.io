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
    values = [0, 1, 2, 3]
    
    # Enumerate all 16 types
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    
    n_types = len(types)  # 16
    n_goods = 2
    
    # Get probabilities from distribution_spec
    if isinstance(distribution_spec, dict) and 'probabilities' in distribution_spec:
        probs = np.array(distribution_spec['probabilities'])
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    else:
        # Default: uniform distribution
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()  # normalize
    
    # Variable ordering:
    # x_00, x_01, ..., x_15,1  (32 vars)
    # p_0, p_1, ..., p_15      (16 vars)
    # Total 48 vars.
    
    n_vars = n_types * n_goods + n_types
    x_offset = 0
    p_offset = n_types * n_goods
    
    # Objective: Maximize sum(pi_i * p_i)
    # linprog minimizes, so use -pi_i
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_offset + i] = -probs[i]
        
    # Constraints: A_ub x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # 1. IR Constraints: p_i <= v_i . x_i
    # p_i - v_i . x_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[p_offset + i] = 1.0
        for j in range(n_goods):
            v_ij = types[i][j]
            row[x_offset + i * n_goods + j] = -v_ij
        A_rows.append(row)
        b_rows.append(0.0)
        
    # 2. IC Constraints: v_i . x_i - p_i >= v_k . x_k - p_k
    # => p_i - v_i . x_i <= p_k - v_k . x_k
    # => p_i - p_k - v_i . x_i + v_k . x_k <= 0
    for i in range(n_types):
        for k in range(n_types):
            if i == k:
                continue
            row = np.zeros(n_vars)
            # p_i term
            row[p_offset + i] = 1.0
            # -p_k term
            row[p_offset + k] = -1.0
            # -v_i . x_i terms
            for j in range(n_goods):
                v_ij = types[i][j]
                row[x_offset + i * n_goods + j] = -v_ij
            # +v_k . x_k terms
            for j in range(n_goods):
                v_kj = types[k][j]
                row[x_offset + k * n_goods + j] = v_kj
            A_rows.append(row)
            b_rows.append(0.0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds:
    # x in [0, 1]
    # p in [0, None] (IR ensures p <= utility, but we need p >= 0)
    bounds = []
    for i in range(n_types * n_goods):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status != 0:
        # Fallback to simple threshold if LP fails (should not happen)
        # Re-use simple logic from before as fallback
        best_revenue = -1
        best_r1, best_r2 = 0, 0
        for r1 in values:
            for r2 in values:
                alloc = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                for idx, (v1, v2) in enumerate(types):
                    x1 = 1.0 if v1 >= r1 else 0.0
                    x2 = 1.0 if v2 >= r2 else 0.0
                    alloc[idx, 0] = x1
                    alloc[idx, 1] = x2
                    p1 = v1 * x1 - max(0, v1 - r1)
                    p2 = v2 * x2 - max(0, v2 - r2)
                    payments[idx] = p1 + p2
                if np.all(payments >= -1e-10):
                    revenue = np.sum(probs * payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_r1, best_r2 = r1, r2
                        
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
        payments = np.maximum(payments, 0)
        return {
            'allocations': alloc,
            'payments': payments
        }
        
    # Extract solution
    x_flat = res.x[x_offset : x_offset + n_types * n_goods]
    p_flat = res.x[p_offset : p_offset + n_types]
    
    alloc = x_flat.reshape((n_types, n_goods))
    payments = p_flat
    
    # Clean up numerical noise
    alloc = np.clip(alloc, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify IC/IR just in case (optional but good for debugging)
    # The LP guarantees this, but floating point might be tight.
    
    return {
        'allocations': alloc,
        'payments': payments
    }
# EVOLVE-BLOCK-END
