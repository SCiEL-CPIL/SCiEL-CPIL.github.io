# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: The support of possible values
        constraints: Additional constraints
        
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    
    # Parse the distribution
    probs = _parse_distribution(distribution_spec, support, constraints)
    n_types = 16
    n_goods = 2
    
    # Types are indexed as (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Solve the optimal mechanism via LP
    allocations, payments = _solve_optimal_mechanism(probs, types, n_types, n_goods)
    
    # Verify the mechanism
    _verify_mechanism(allocations, payments, probs, types)
    
    return {
        'allocations': allocations,
        'payments': payments
    }

def _parse_distribution(distribution_spec, support, constraints):
    """
    Parse the distribution specification into a probability vector over 16 types.
    """
    n_types = 16
    
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        dist_type = distribution_spec.get('type', 'uniform')
        
        if dist_type == 'uniform':
            return np.ones(n_types) / n_types
        
        elif dist_type == 'product':
            marginals = distribution_spec.get('marginals', None)
            if marginals is not None and len(marginals) == 2:
                p1, p2 = marginals
                probs = np.zeros(n_types)
                for v1 in range(4):
                    for v2 in range(4):
                        idx = v1 * 4 + v2
                        probs[idx] = p1[v1] * p2[v2]
                s = probs.sum()
                if s > 1e-12:
                    return probs / s
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        
        elif dist_type == 'explicit':
            probs_vec = distribution_spec.get('probs', None)
            if probs_vec is not None:
                probs = np.array(probs_vec, dtype=float)
                if len(probs) == n_types:
                    s = probs.sum()
                    if s > 1e-12:
                        return probs / s
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        
        else:
            return np.ones(n_types) / n_types
    
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == n_types:
            s = probs.sum()
            if s > 1e-12:
                return probs / s
        return np.ones(n_types) / n_types
    
    return np.ones(n_types) / n_types

def _solve_optimal_mechanism(probs, types, n_types, n_goods):
    """
    Solve for the optimal incentive-compatible mechanism via LP.
    
    Variables:
    - x[i][j] for i in 0..15, j in 0..1: allocation of good j to type i
    - p[i] for i in 0..15: payment by type i
    
    Total: 32 + 16 = 48 variables
    
    Objective: Maximize sum_i probs[i] * p[i]
    
    Constraints:
    - IC: For all i, i' != i:
      v_i . x[i] - p[i] >= v_i . x[i'] - p[i']
      => v_i . (x[i] - x[i']) - (p[i] - p[i']) >= 0
      => -v_i . x[i] + v_i . x[i'] + p[i] - p[i'] <= 0
    
    - IR: For all i:
      v_i . x[i] - p[i] >= 0
      => -v_i . x[i] + p[i] <= 0
    
    - 0 <= x[i][j] <= 1
    - p[i] >= 0
    """
    # Variable layout: [x_00, x_01, x_10, x_11, ..., x_150, x_151, p_0, p_1, ..., p_15]
    n_x = n_types * n_goods  # 32
    n_p = n_types            # 16
    n_vars = n_x + n_p       # 48
    
    # Objective: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_x + i] = -probs[i]
    
    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: 16*15 = 240 constraints
    for i in range(n_types):
        v_i = np.array(types[i], dtype=float)
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            
            row = np.zeros(n_vars)
            
            # -v_i . x[i]
            for k in range(n_goods):
                row[i * n_goods + k] = -v_i[k]
            
            # +v_i . x[i']
            for k in range(n_goods):
                row[i_prime * n_goods + k] = v_i[k]
            
            # +p[i] - p[i']
            row[n_x + i] = 1.0
            row[n_x + i_prime] = -1.0
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: 16 constraints
    for i in range(n_types):
        v_i = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        for k in range(n_goods):
            row[i * n_goods + k] = -v_i[k]
        row[n_x + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Upper bound on allocations: x[i][j] <= 1
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(n_vars)
            row[i * n_goods + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= x[i][j] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP with HiGHS
    options = {
        'presolve': True,
        'dual_feasibility_tolerance': 1e-9,
        'primal_feasibility_tolerance': 1e-9,
        'dual_staleness_threshold': 1e-9,
        'primal_staleness_threshold': 1e-9,
        'time_limit': 60.0,
    }
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    
    if not result.success:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i in range(n_types):
            v1, v2 = types[i]
            if v1 >= 2:
                allocations[i, 0] = 1
            if v2 >= 2:
                allocations[i, 1] = 1
            payments[i] = 0
        return allocations, payments
    
    x_flat = result.x[:n_x]
    p_flat = result.x[n_x:]
    
    allocations = x_flat.reshape(n_types, n_goods)
    payments = p_flat
    
    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return allocations, payments

def _verify_mechanism(allocations, payments, probs, types):
    """
    Verify that the mechanism is feasible, IR, and IC.
    """
    n_types = len(types)
    n_goods = 2
    
    # Check feasibility
    assert np.all(allocations >= -1e-9), "Allocations must be non-negative"
    assert np.all(allocations <= 1 + 1e-9), "Allocations must be at most 1"
    assert np.all(payments >= -1e-9), "Payments must be non-negative"
    
    # Check IR
    for i in range(n_types):
        v_i = types[i]
        utility = sum(v_i[k] * allocations[i, k] for k in range(n_goods)) - payments[i]
        assert utility >= -1e-8, f"IR violated for type {i}: utility={utility}"
    
    # Check IC
    for i in range(n_types):
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            v_i = types[i]
            util_report = sum(v_i[k] * allocations[i_prime, k] for k in range(n_goods)) - payments[i_prime]
            util_truth = sum(v_i[k] * allocations[i, k] for k in range(n_goods)) - payments[i]
            regret = util_report - util_truth
            assert regret <= 1e-8, f"IC violated: type {i} reporting {i_prime}, regret={regret}"

# EVOLVE-BLOCK-END
