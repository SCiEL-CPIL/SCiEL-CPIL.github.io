"""Incentive-compatible finite auction design - LP-based solution."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the type distribution
        support: Support of types (values in {0,1,2,3} for 2 goods)
        constraints: Additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = list(product([0, 1, 2, 3], repeat=2))
    n_types = len(types)  # 16 types
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, types)
    
    # Try to solve the LP
    result = _solve_auction_lp(types, probs, n_goods, deadline)
    
    if result is not None:
        return result
    
    # Fallback: baseline mechanism (no allocation, no payment)
    return _baseline_result(n_types)


def _parse_distribution(distribution_spec, types):
    """Parse the distribution specification into probabilities for each type."""
    n_types = len(types)
    
    # Handle various distribution specifications
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    # Check if it's already a probability vector
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        spec = np.array(distribution_spec, dtype=float)
        if spec.shape == (n_types,) and np.isclose(spec.sum(), 1.0, atol=1e-6):
            return spec / spec.sum()
    
    # Check for 'uniform' keyword
    if isinstance(distribution_spec, str):
        spec_lower = distribution_spec.lower()
        if 'uniform' in spec_lower:
            return np.ones(n_types) / n_types
    
    # Try to extract probabilities from dict format
    if isinstance(distribution_spec, dict):
        # Could be {type: prob} format
        probs = np.zeros(n_types)
        type_map = {t: i for i, t in enumerate(types)}
        for t, p in distribution_spec.items():
            if t in type_map:
                probs[type_map[t]] = p
        if probs.sum() > 0:
            return probs / probs.sum()
        
        # Could have 'probs' or 'probabilities' key
        for key in ['probs', 'probabilities', 'weights']:
            if key in distribution_spec:
                spec = np.array(distribution_spec[key], dtype=float)
                if spec.shape == (n_types,):
                    return spec / spec.sum()
    
    # Default to uniform
    return np.ones(n_types) / n_types


def _solve_auction_lp(types, probs, n_goods, deadline):
    """
    Solve the LP to maximize expected revenue subject to IC and IR constraints.
    Uses the envelope theorem formulation with utilities u_i.
    """
    from scipy.optimize import linprog
    
    n_types = len(types)
    
    # Variable indexing:
    # 0 .. n_types*n_goods - 1 : x_{i,k}
    # n_types*n_goods .. 2*n_types*n_goods - 1 : u_i
    n_x = n_types * n_goods
    n_u = n_types
    n_vars = n_x + n_u
    
    def var_x(i, k):
        return i * n_goods + k

    def var_u(i):
        return n_x + i
    
    # Objective: Maximize sum_t probs[t] * (v_t . x_t - u_t)
    # Minimize -sum_t probs[t] * (v_t . x_t - u_t)
    # = sum_t probs[t] * u_t - sum_t probs[t] * v_t . x_t
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[var_u(t)] = probs[t]
        for g in range(n_goods):
            c[var_x(t, g)] = -probs[t] * types[t][g]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC Constraints: u_i - u_j + (v_j - v_i) . x_i <= 0 for all i != j
    # This ensures type j reporting i gets utility <= u_j
    # Utility of j reporting i: v_j . x_i - p_i
    # p_i = v_i . x_i - u_i
    # Util = v_j . x_i - v_i . x_i + u_i = u_i + (v_j - v_i) . x_i
    # Require u_j >= u_i + (v_j - v_i) . x_i
    # => u_i - u_j + (v_j - v_i) . x_i <= 0
    
    for i in range(n_types):
        v_i = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            v_j = np.array(types[j], dtype=float)
            
            row = np.zeros(n_vars)
            
            # u_i coeff: 1
            row[var_u(i)] = 1.0
            # u_j coeff: -1
            row[var_u(j)] = -1.0
            
            # x_i coeff: v_j - v_i
            for g in range(n_goods):
                row[var_x(i, g)] = v_j[g] - v_i[g]
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    # IR Constraints: u_i >= 0
    # This is handled by bounds, but if we wanted to be explicit in A_ub:
    # -u_i <= 0. But bounds are more efficient.
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = None
        b_ub = None
    
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    if time.time() > deadline - 5.0:
        return None
    
    try:
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='HiGHS',
            options={'time_limit': 70.0, 'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
        )
    except Exception as e:
        return None
    
    if time.time() > deadline:
        return None
    
    if res.status != 0:
        return None
    
    x = res.x
    
    allocations = np.zeros((n_types, n_goods))
    utilities = np.zeros(n_types)
    
    for t in range(n_types):
        for g in range(n_goods):
            allocations[t, g] = max(0.0, x[var_x(t, g)])
        utilities[t] = max(0.0, x[var_u(t)])
    
    # Calculate payments
    payments = np.zeros(n_types)
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        payments[t] = v_t @ allocations[t] - utilities[t]
        # Clip small negative values due to numerical error
        if payments[t] < 0 and payments[t] > -1e-8:
            payments[t] = 0.0
    
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    # Verification
    if _verify_mechanism(types, probs, allocations, payments):
        # Optional: Polish step if time permits
        # Since the problem is small, a second solve with tighter tolerance might help
        # but the first solve is usually exact enough for 1e-8.
        # We will just return the verified result.
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    return None


def _verify_mechanism(types, probs, allocations, payments):
    """Verify that the mechanism satisfies all constraints."""
    n_types = len(types)
    eps = 1e-8
    
    # Check feasibility
    if np.any(allocations < -eps) or np.any(allocations > 1 + eps):
        return False
    if np.any(payments < -eps):
        return False
    
    # Check IC: For each type t, truth-telling utility >= misreporting utility
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_truth = v_t @ allocations[t] - payments[t]
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            util_misreport = v_t @ allocations[t_prime] - payments[t_prime]
            if util_misreport > util_truth + eps:
                return False
    
    # Check IR: Utility >= 0 for all types
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        if util_t < -eps:
            return False
    
    return True


def _baseline_result(n_types):
    """Return a simple baseline mechanism that is always feasible."""
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
