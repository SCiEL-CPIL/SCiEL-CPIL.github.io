"""Incentive-compatible finite auction design via direct LP solve with tight tolerances."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse support: 16 reports, each (v1, v2) with v1,v2 in {0,1,2,3}
    if support is None:
        support = [(i, j) for i in range(4) for j in range(4)]
    n = len(support)
    assert n == 16, f"Expected 16 reports, got {n}"
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.ones(n) / n
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n) / n
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
    
    # Values: for each report i, values are (v1_i, v2_i)
    values = np.array(support, dtype=float)  # shape (16, 2)
    
    # LP formulation for single-bidder truthful auction
    # Variables: x_i^1, x_i^2 (allocations), p_i (payments) for each report i
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize sum_i probs[i] * p_i
    # c = [0,0,...,0 (32 zeros), probs[0], probs[1], ..., probs[15]]
    c = np.zeros(48)
    c[32:] = -probs  # minimize -revenue
    
    # Constraints:
    # 1. Feasibility: 0 <= x_i^g <= 1 for all i, g
    # 2. Individual rationality: p_i <= sum_g x_i^g * v_i^g for all i
    # 3. Incentive compatibility: for all i, j:
    #    p_i - p_j <= sum_g x_j^g * (v_i^g - v_j^g)
    #    (report i is weakly preferred to report j when true type is i)
    
    # Build A_ub and b_ub for linprog (minimize c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq)
    
    # IR constraints: p_i - sum_g x_i^g * v_i^g >= 0
    # => -p_i + sum_g x_i^g * v_i^g <= 0
    # For each i: row has -values[i,0] at pos i*2, -values[i,1] at pos i*2+1, -1 at pos 32+i
    
    A_rows = []
    b_rows = []
    
    # IR constraints (16)
    for i in range(n):
        row = np.zeros(48)
        row[i*2] = -values[i, 0]
        row[i*2 + 1] = -values[i, 1]
        row[32 + i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: for all pairs (i, j), i != j:
    # U(i|i) >= U(i|j) where U(type, report) = sum_g x_report^g * v_type^g - p_report
    # => sum_g x_j^g * v_i^g - p_i >= sum_g x_j^g * v_j^g - p_j
    # => -sum_g x_j^g * (v_i^g - v_j^g) + p_j - p_i >= 0
    # => sum_g x_j^g * (v_i^g - v_j^g) - p_j + p_i <= 0
    # Wait, let me redo this carefully.
    # 
    # Utility of type i reporting j: U(i, j) = sum_g x_j^g * v_i^g - p_j
    # Truthfulness: U(i, i) >= U(i, j) for all i, j
    # sum_g x_i^g * v_i^g - p_i >= sum_g x_j^g * v_i^g - p_j
    # => -sum_g x_i^g * v_i^g + p_i + sum_g x_j^g * v_i^g - p_j <= 0
    # => sum_g (x_j^g - x_i^g) * v_i^g + p_i - p_j <= 0
    #
    # For each pair (i, j) with i != j:
    # Coefficients:
    #   For variable x_i^g (pos i*2+g): -v_i^g
    #   For variable x_j^g (pos j*2+g): +v_i^g
    #   For variable p_i (pos 32+i): +1
    #   For variable p_j (pos 32+j): -1
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(48)
            for g in range(2):
                row[i*2 + g] = -values[i, g]
                row[j*2 + g] = values[i, g]
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # Non-negativity: x_i^g >= 0, p_i >= 0 (handled by bounds)
    # Upper bounds: x_i^g <= 1
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x_i^g <= 1, 0 <= p_i <= inf
    bounds = []
    for i in range(n):
        for g in range(2):
            bounds.append((0, 1))
    for i in range(n):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: zero allocations, zero payments
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = x[:32].reshape(n, 2)
    payments = x[32:48]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
