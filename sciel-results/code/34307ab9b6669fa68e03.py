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
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Parse the distribution
    probs = _parse_distribution(distribution_spec, support, constraints)
    n_types = 16
    
    # Types are indexed as (v1, v2) where v1, v2 in {0,1,2,3}
    # Index = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Solve the optimal mechanism via LP
    allocations, payments, revenue = _solve_optimal_mechanism(probs, types)
    
    # Verify the mechanism
    _verify_mechanism(allocations, payments, probs, types)
    
    return {
        'allocations': allocations,
        'payments': payments
    }

def _parse_distribution(distribution_spec, support, constraints):
    """
    Parse the distribution specification into a probability vector over 16 types.
    Types are (v1, v2) with v1, v2 in {0,1,2,3}.
    """
    n_types = 16
    
    # Try to extract probabilities from distribution_spec
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        dist_type = distribution_spec.get('type', 'uniform')
        
        if dist_type == 'uniform':
            return np.ones(n_types) / n_types
        
        elif dist_type == 'product':
            # Product of marginals over each good
            marginals = distribution_spec.get('marginals', None)
            if marginals is not None and len(marginals) == 2:
                p1, p2 = marginals
                # p1 is distribution over v1, p2 over v2
                probs = np.zeros(n_types)
                for v1 in range(4):
                    for v2 in range(4):
                        idx = v1 * 4 + v2
                        probs[idx] = p1[v1] * p2[v2]
                return probs
            else:
                return np.ones(n_types) / n_types
        
        elif dist_type == 'explicit':
            # Explicit probability vector
            probs_vec = distribution_spec.get('probs', None)
            if probs_vec is not None:
                probs = np.array(probs_vec, dtype=float)
                if len(probs) == n_types:
                    probs = probs / probs.sum()
                    return probs
                else:
                    return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        
        else:
            return np.ones(n_types) / n_types
    
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == n_types:
            probs = probs / probs.sum()
            return probs
        else:
            return np.ones(n_types) / n_types
    
    # Default to uniform
    return np.ones(n_types) / n_types

def _solve_optimal_mechanism(probs, types):
    """
    Solve for the optimal incentive-compatible mechanism.
    
    For a single additive bidder, we can use the following approach:
    - The allocation rule x(t) must be monotone: if t1 >= t2 componentwise, then x(t1) >= x(t2)
    - Payments are determined by the envelope theorem
    - We maximize expected revenue subject to IC and IR
    
    Variables:
    - x[i][j] for i in 0..15, j in 0..1: allocation of good j to type i
    - p[i] for i in 0..15: payment by type i
    
    Constraints:
    - IC: For all i, i', sum_k v_i[k]*x[i'][k] - p[i'] <= sum_k v_i[k]*x[i][k] - p[i]
      (Type i prefers reporting truthfully to reporting i')
    - IR: For all i, sum_k v_i[k]*x[i][k] - p[i] >= 0
    - Feasibility: 0 <= x[i][j] <= 1
    - p[i] >= 0
    
    Objective: Maximize sum_i probs[i] * p[i]
    """
    n_types = len(types)
    n_goods = 2
    
    # Decision variables:
    # x[0..15][0..1] (32 variables), p[0..15] (16 variables) = 48 variables
    # Total: 48 variables
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog: minimize -sum_i probs[i] * p[i]
    c = np.zeros(48)
    for i in range(n_types):
        c[32 + i] = -probs[i]  # Negative because we minimize
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub = []
    b_ub = []
    
    # IC constraints: For each pair (i, i'), type i's utility from reporting i 
    # must be >= utility from reporting i'
    # u(i, i') = sum_k v_i[k]*x[i'][k] - p[i']
    # u(i, i) = sum_k v_i[k]*x[i][k] - p[i]
    # IC: u(i, i) >= u(i, i') for all i, i'
    # => sum_k v_i[k]*x[i][k] - p[i] >= sum_k v_i[k]*x[i'][k] - p[i']
    # => sum_k v_i[k]*x[i][k] - sum_k v_i[k]*x[i'][k] - p[i] + p[i'] >= 0
    # => -sum_k v_i[k]*x[i][k] + sum_k v_i[k]*x[i'][k] + p[i] - p[i'] <= 0
    
    for i in range(n_types):
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            
            row = np.zeros(48)
            v_i = types[i]  # (v1, v2)
            
            # -sum_k v_i[k]*x[i][k]
            for k in range(n_goods):
                row[i * n_goods + k] = -v_i[k]
            
            # +sum_k v_i[k]*x[i'][k]
            for k in range(n_goods):
                row[i_prime * n_goods + k] = v_i[k]
            
            # +p[i] - p[i']
            row[32 + i] = 1
            row[32 + i_prime] = -1
            
            A_ub.append(row)
            b_ub.append(0)
    
    # IR constraints: For each i, sum_k v_i[k]*x[i][k] - p[i] >= 0
    # => -sum_k v_i[k]*x[i][k] + p[i] <= 0
    
    for i in range(n_types):
        row = np.zeros(48)
        v_i = types[i]
        for k in range(n_goods):
            row[i * n_goods + k] = -v_i[k]
        row[32 + i] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # Feasibility: 0 <= x[i][j] <= 1
    # x[i][j] >= 0 is handled by bounds
    # x[i][j] <= 1: x[i][j] <= 1
    for i in range(n_types):
        for j in range(n_goods):
            row = np.zeros(48)
            row[i * n_goods + j] = 1
            A_ub.append(row)
            b_ub.append(1)
    
    # p[i] >= 0 is handled by bounds
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x[i][j] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i in range(n_types):
            v1, v2 = types[i]
            # Allocate good 1 if v1 >= 2, good 2 if v2 >= 2
            if v1 >= 2:
                allocations[i, 0] = 1
            if v2 >= 2:
                allocations[i, 1] = 1
            payments[i] = 0
        return allocations, payments, 0.0
    
    x_flat = result.x[:32]
    p_flat = result.x[32:48]
    
    allocations = x_flat.reshape(n_types, n_goods)
    payments = p_flat
    
    # Ensure non-negativity
    allocations = np.maximum(allocations, 0)
    allocations = np.minimum(allocations, 1)
    payments = np.maximum(payments, 0)
    
    revenue = np.sum(probs * payments)
    
    return allocations, payments, revenue

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
            # Utility of type i when reporting i_prime
            util_report = sum(v_i[k] * allocations[i_prime, k] for k in range(n_goods)) - payments[i_prime]
            # Utility of type i when reporting truthfully
            util_truth = sum(v_i[k] * allocations[i, k] for k in range(n_goods)) - payments[i]
            regret = util_report - util_truth
            assert regret <= 1e-8, f"IC violated: type {i} reporting {i_prime}, regret={regret}"

# EVOLVE-BLOCK-END
