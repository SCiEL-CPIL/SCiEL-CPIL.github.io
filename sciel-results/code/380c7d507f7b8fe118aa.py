"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution_spec to get probabilities for each type
    # distribution_spec could be a dict, list, or other format
    # Let's handle common formats
    if isinstance(distribution_spec, dict):
        # Try to get probabilities from dict
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
            if len(probs) != 16:
                # Try to normalize or handle differently
                probs = np.ones(16) / 16.0
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure probabilities are valid
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    
    # Decision variables:
    # x_{j,t} for j in {0,1}, t in 0..15: 32 variables
    # p_t for t in 0..15: 16 variables
    # Total: 48 variables
    # Indexing: var_idx = j*16 + t for x_{j,t}, var_idx = 32 + t for p_t
    
    n_vars = 48
    
    # Objective: maximize sum_t probs[t] * p_t
    # linprog minimizes, so we minimize -sum_t probs[t] * p_t
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[32 + t] = -probs[t]
    
    # Constraints:
    # 1. IC constraints: for all t (true type) and r (report):
    #    sum_j x_{j,r} * v_j(t) - p_r <= sum_j x_{j,t} * v_j(t) - p_t
    #    Rearranged: sum_j x_{j,r} * v_j(t) - sum_j x_{j,t} * v_j(t) - p_r + p_t <= 0
    #    Or: sum_j [v_j(t)*x_{j,r} - v_j(t)*x_{j,t}] - p_r + p_t <= 0
    
    A_ub = []
    b_ub = []
    
    for t in range(n_types):
        for r in range(n_types):
            if t == r:
                continue  # IC is trivially satisfied when t=r
            row = np.zeros(n_vars)
            for j in range(2):
                v_j_t = type_values[t, j]
                # x_{j,r} coefficient: v_j_t
                row[j * 16 + r] += v_j_t
                # x_{j,t} coefficient: -v_j_t
                row[j * 16 + t] -= v_j_t
                # p_r coefficient: -1
                row[32 + r] -= 1
                # p_t coefficient: +1
                row[32 + t] += 1
            A_ub.append(row)
            b_ub.append(0.0)
    
    # 2. IR constraints: for all t:
    #    sum_j x_{j,t} * v_j(t) - p_t >= 0
    #    => -sum_j x_{j,t} * v_j(t) + p_t <= 0
    
    for t in range(n_types):
        row = np.zeros(n_vars)
        for j in range(2):
            v_j_t = type_values[t, j]
            row[j * 16 + t] -= v_j_t
            row[32 + t] += 1
        A_ub.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x_{j,t} <= 1, p_t >= 0 (no upper bound on payments)
    bounds = []
    for j in range(2):
        for t in range(n_types):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a simple valid mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        allocations = np.zeros((n_types, 2))
        for t in range(n_types):
            for j in range(2):
                allocations[t, j] = x[j * 16 + t]
        payments = x[32:32 + n_types]
        
        # Ensure non-negative
        allocations = np.maximum(allocations, 0)
        allocations = np.minimum(allocations, 1)
        payments = np.maximum(payments, 0)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback: return a simple valid mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
