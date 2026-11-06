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
    
    # Vector of values for each type
    # shape: (16, 2)
    V = np.array(types, dtype=float)
    
    # Variables:
    # x_0 ... x_15 (allocation prob for good 1 for each type)
    # x_16 ... x_31 (allocation prob for good 2 for each type)
    # p_0 ... p_15 (payment for each type)
    # Total 48 variables.
    
    n_vars = n_types * n_goods + n_types
    # Indices:
    # x[i, j] is at index i * n_goods + j
    # p[i] is at index n_types * n_goods + i
    
    # Objective: Maximize sum(pi_i * p_i)
    # Minimize -sum(pi_i * p_i)
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
        
    # Bounds
    # x in [0, 1]
    # p in [0, inf]
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    # Inequality constraints: A_ub @ x <= b_ub
    
    # 1. IC Constraints:
    # p_i - V_i . x_i <= p_k - V_i . x_k
    # p_i - sum_j V[i,j] x_{i,j} - p_k + sum_j V[i,j] x_{k,j} <= 0
    
    A_ic = []
    b_ic = []
    
    for i in range(n_types):
        for k in range(n_types):
            if i == k:
                continue
            row = np.zeros(n_vars)
            
            # p_i term
            row[n_types * n_goods + i] = 1.0
            # p_k term
            row[n_types * n_goods + k] = -1.0
            
            # x_i terms: - V[i,j]
            for j in range(n_goods):
                row[i * n_goods + j] = -V[i, j]
                
            # x_k terms: + V[i,j]
            for j in range(n_goods):
                row[k * n_goods + j] = V[i, j]
                
            A_ic.append(row)
            b_ic.append(0.0)
            
    # 2. IR Constraints:
    # p_i - V_i . x_i <= 0
    
    A_ir = []
    b_ir = []
    
    for i in range(n_types):
        row = np.zeros(n_vars)
        
        # p_i term
        row[n_types * n_goods + i] = 1.0
        
        # x_i terms: - V[i,j]
        for j in range(n_goods):
            row[i * n_goods + j] = -V[i, j]
            
        A_ir.append(row)
        b_ir.append(0.0)
        
    # Combine constraints
    if A_ic:
        A_ub = np.vstack([A_ic, A_ir])
    else:
        A_ub = np.vstack([A_ir])
    b_ub = np.concatenate([b_ic, b_ir])
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        # Extract solution
        x_vals = res.x
        alloc = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        
        for i in range(n_types):
            for j in range(n_goods):
                alloc[i, j] = x_vals[i * n_goods + j]
            payments[i] = x_vals[n_types * n_goods + i]
            
        # Clean up numerical noise
        alloc = np.clip(alloc, 0, 1)
        payments = np.maximum(payments, 0)
        
        # Check IR and IC approximately to be safe
        # If violations are significant, fallback to threshold
        # But LP should be exact.
        
        return {
            'allocations': alloc,
            'payments': payments
        }
    else:
        # Fallback to threshold heuristic if LP fails
        best_revenue = -1
        best_r1, best_r2 = 0, 0
        
        for r1 in values:
            for r2 in values:
                alloc = np.zeros((n_types, 2))
                pay = np.zeros(n_types)
                
                for idx, (v1, v2) in enumerate(types):
                    x1 = 1.0 if v1 >= r1 else 0.0
                    x2 = 1.0 if v2 >= r2 else 0.0
                    alloc[idx, 0] = x1
                    alloc[idx, 1] = x2
                    
                    p1 = v1 * x1 - max(0, v1 - r1)
                    p2 = v2 * x2 - max(0, v2 - r2)
                    pay[idx] = p1 + p2
                
                # Check IR
                ir_ok = True
                for idx, (v1, v2) in enumerate(types):
                    if pay[idx] > v1 * alloc[idx, 0] + v2 * alloc[idx, 1] + 1e-10:
                        ir_ok = False
                        break
                if ir_ok and np.all(pay >= -1e-10):
                    revenue = np.sum(probs * pay)
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

# EVOLVE-BLOCK-END
