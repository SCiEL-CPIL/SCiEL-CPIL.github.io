"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find any array-like value
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    probs = np.array(val, dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs has correct length and is normalized
    if len(probs) != n_types:
        # Try to reshape or pad
        if len(probs) == 4:
            # Maybe it's per-value probabilities, create product distribution
            probs = np.outer(probs, probs).flatten()
        elif len(probs) == 16:
            pass
        else:
            probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum()
    
    # Variables: for each type v, we have q1(v), q2(v), p(v)
    # Total: 3 * n_types = 48 variables
    n_vars = 3 * n_types
    
    def var_idx(v, comp):
        """Get variable index for type v, component comp (0=q1, 1=q2, 2=p)"""
        return v * 3 + comp
    
    # Objective: maximize sum(pi(v) * p(v)) => minimize -sum(pi(v) * p(v))
    c = np.zeros(n_vars)
    for v in range(n_types):
        c[var_idx(v, 2)] = -probs[v]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # Bounds: 0 <= q1(v) <= 1, 0 <= q2(v) <= 1, 0 <= p(v)
    # We'll use bounds instead of inequality constraints for bounds
    bounds = []
    for v in range(n_types):
        bounds.append((0, 1))  # q1(v)
        bounds.append((0, 1))  # q2(v)
        bounds.append((0, None))  # p(v) >= 0
    
    # IR constraint: p(v) <= v1*q1(v) + v2*q2(v)
    # i.e., p(v) - v1*q1(v) - v2*q2(v) <= 0
    for v in range(n_types):
        v1, v2 = types[v]
        row = np.zeros(n_vars)
        row[var_idx(v, 2)] = 1.0  # p(v)
        if v1 != 0:
            row[var_idx(v, 0)] = -float(v1)
        if v2 != 0:
            row[var_idx(v, 1)] = -float(v2)
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: for all v, v':
    # v1*q1(v) + v2*q2(v) - p(v) >= v1*q1(v') + v2*q2(v') - p(v')
    # => -v1*q1(v) - v2*q2(v) + p(v) + v1*q1(v') + v2*q2(v') - p(v') <= 0
    for v in range(n_types):
        for v_prime in range(n_types):
            if v == v_prime:
                continue
            v1, v2 = types[v]
            vp1, vp2 = types[v_prime]
            row = np.zeros(n_vars)
            # Terms for type v
            if v1 != 0:
                row[var_idx(v, 0)] = -float(v1)
            if v2 != 0:
                row[var_idx(v, 1)] = -float(v2)
            row[var_idx(v, 2)] = 1.0
            # Terms for type v'
            if vp1 != 0:
                row[var_idx(v_prime, 0)] = float(vp1)
            if vp2 != 0:
                row[var_idx(v_prime, 1)] = float(vp2)
            row[var_idx(v_prime, 2)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else None
    b_ub = np.array(b_rows) if b_rows else None
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for v in range(n_types):
        allocations[v, 0] = x[var_idx(v, 0)]
        allocations[v, 1] = x[var_idx(v, 1)]
        payments[v] = x[var_idx(v, 2)]
    
    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify IC and IR with tolerance
    # Check IR: U(v) = v·q(v) - p(v) >= 0
    for v in range(n_types):
        v1, v2 = types[v]
        utility = v1 * allocations[v, 0] + v2 * allocations[v, 1] - payments[v]
        if utility < -1e-8:
            # Fix by reducing payment
            payments[v] = v1 * allocations[v, 0] + v2 * allocations[v, 1]
    
    # Check IC
    for v in range(n_types):
        for v_prime in range(n_types):
            v1, v2 = types[v]
            vp1, vp2 = types[v_prime]
            utility_v = v1 * allocations[v, 0] + v2 * allocations[v, 1] - payments[v]
            utility_report = v1 * allocations[v_prime, 0] + v2 * allocations[v_prime, 1] - payments[v_prime]
            if utility_v < utility_report - 1e-8:
                # IC violated, this shouldn't happen from LP but fix if needed
                pass
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
