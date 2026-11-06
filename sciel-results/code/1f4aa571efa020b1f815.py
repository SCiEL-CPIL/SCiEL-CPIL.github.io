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
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
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
    Uses reduced constraint set (adjacent types + monotonicity) for speed.
    """
    from scipy.optimize import linprog
    
    n_types = len(types)
    n_goods = 2
    
    def var_alloc(t, g):
        return t * n_goods + g
    
    def var_pay(t):
        return n_types * n_goods + t
    
    n_vars = n_types * n_goods + n_types
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[var_pay(t)] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Identify adjacent types in the lattice
    # types are (v1, v2)
    adj_pairs = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if i == j: continue
            diff = [t1[k] - t2[k] for k in range(n_goods)]
            # Check if they differ by exactly 1 in one coordinate and 0 in others
            non_zero = [d for d in diff if d != 0]
            if len(non_zero) == 1 and abs(non_zero[0]) == 1:
                adj_pairs.append((i, j))
    
    # IC constraints for adjacent pairs
    # v_t @ x[t] - p[t] >= v_t @ x[t'] - p[t']
    # => v_t @ x[t] - v_t @ x[t'] - p[t] + p[t'] >= 0
    # => - (v_t @ x[t] - v_t @ x[t'] - p[t] + p[t']) <= 0
    for t_idx, t_prime_idx in adj_pairs:
        t = types[t_idx]
        t_prime = types[t_prime_idx]
        v_t = np.array(t, dtype=float)
        
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[var_alloc(t_idx, g)] += v_t[g]
            row[var_alloc(t_prime_idx, g)] -= v_t[g]
        row[var_pay(t_idx)] -= 1.0
        row[var_pay(t_prime_idx)] += 1.0
        
        A_ub_rows.append(-row)
        b_ub_vals.append(0.0)
        
        # Also add the reverse direction? 
        # IC requires U(t) >= U(t|t') AND U(t') >= U(t'|t)
        # The reverse is: v_t' @ x[t'] - p[t'] >= v_t' @ x[t] - p[t]
        v_tp = np.array(t_prime, dtype=float)
        row_rev = np.zeros(n_vars)
        for g in range(n_goods):
            row_rev[var_alloc(t_prime_idx, g)] += v_tp[g]
            row_rev[var_alloc(t_idx, g)] -= v_tp[g]
        row_rev[var_pay(t_prime_idx)] -= 1.0
        row_rev[var_pay(t_idx)] += 1.0
        
        A_ub_rows.append(-row_rev)
        b_ub_vals.append(0.0)

    # Monotonicity constraints: x_t >= x_t' if t > t'
    # We add these to ensure the local IC implies global IC
    for t_idx, t_prime_idx in adj_pairs:
        t = types[t_idx]
        t_prime = types[t_prime_idx]
        # Check if t > t' (component-wise)
        if all(t[g] >= t_prime[g] for g in range(n_goods)) and any(t[g] > t_prime[g] for g in range(n_goods)):
            for g in range(n_goods):
                row = np.zeros(n_vars)
                row[var_alloc(t_idx, g)] = 1.0
                row[var_alloc(t_prime_idx, g)] = -1.0
                A_ub_rows.append(row) # x_t - x_t' >= 0 => x_t - x_t' <= 1? No, x_t >= x_t' => -x_t + x_t' <= 0
                # Wait, linprog is A_ub x <= b_ub.
                # We want x_t >= x_t' => x_t - x_t' >= 0 => -(x_t - x_t') <= 0
                A_ub_rows[-1] = -row
                b_ub_vals.append(0.0)
        
        # Check if t' > t
        if all(t_prime[g] >= t[g] for g in range(n_goods)) and any(t_prime[g] > t[g] for g in range(n_goods)):
            for g in range(n_goods):
                row = np.zeros(n_vars)
                row[var_alloc(t_prime_idx, g)] = 1.0
                row[var_alloc(t_idx, g)] = -1.0
                A_ub_rows.append(-row)
                b_ub_vals.append(0.0)

    # IR constraints
    for t_idx in range(n_types):
        v_t = np.array(types[t_idx], dtype=float)
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[var_alloc(t_idx, g)] = v_t[g]
        row[var_pay(t_idx)] = -1.0
        A_ub_rows.append(-row)
        b_ub_vals.append(0.0)
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = None
        b_ub = None
    
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
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
            options={'time_limit': 70.0}
        )
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    if res.status != 0:
        return None
    
    x = res.x
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        for g in range(n_goods):
            allocations[t, g] = max(0.0, x[var_alloc(t, g)])
        payments[t] = max(0.0, x[var_pay(t)])
    
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    if _verify_mechanism(types, probs, allocations, payments):
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
    # Allocate nothing, charge nothing - always IR and IC
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
