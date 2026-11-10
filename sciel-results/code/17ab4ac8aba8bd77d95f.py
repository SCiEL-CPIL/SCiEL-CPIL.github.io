"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def _parse_distribution(distribution_spec, n_types, types):
    """Parse distribution spec into a probability vector of length n_types."""
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float).flatten()
            if len(probs) == n_types:
                probs = np.maximum(probs, 0)
                s = probs.sum()
                if s > 0:
                    return probs / s
                return np.ones(n_types) / n_types
            return np.ones(n_types) / n_types
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if dist == 'uniform':
                return np.ones(n_types) / n_types
            elif dist == 'geometric':
                # P(v1=k, v2=l) proportional to 0.7^k * 0.7^l
                probs = np.array([0.7**v1 * 0.7**v2 for v1, v2 in types])
                probs /= probs.sum()
                return probs
            elif dist == 'independent':
                # Try to parse independent marginals
                if 'p1' in distribution_spec and 'p2' in distribution_spec:
                    p1 = np.array(distribution_spec['p1'], dtype=float)
                    p2 = np.array(distribution_spec['p2'], dtype=float)
                    probs = np.zeros(n_types)
                    for idx, (v1, v2) in enumerate(types):
                        probs[idx] = p1[v1] * p2[v2]
                    probs = np.maximum(probs, 0)
                    s = probs.sum()
                    if s > 0:
                        return probs / s
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        else:
            return np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) == n_types:
            probs = np.maximum(probs, 0)
            s = probs.sum()
            if s > 0:
                return probs / s
            return np.ones(n_types) / n_types
        return np.ones(n_types) / n_types
    else:
        return np.ones(n_types) / n_types


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # Generate all 16 types
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)  # 16
    
    # Parse probabilities
    probs = _parse_distribution(distribution_spec, n_types, types)
    
    # Variables:
    # x[i] = probability of allocating good 1 to type i (16 vars)
    # y[i] = probability of allocating good 2 to type i (16 vars)
    # p[i] = payment for type i (16 vars)
    # Total: 48 variables
    # Order: [x_0..x_15, y_0..y_15, p_0..p_15]
    
    n_vars = 3 * n_types  # 48
    
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    c[2*n_types:3*n_types] = -probs
    
    # Constraints:
    # 1. IR: For each type i, p[i] <= x[i]*v1_i + y[i]*v2_i
    #    => -x[i]*v1_i - y[i]*v2_i + p[i] <= 0
    # 2. IC: For each true type i and report j,
    #    x[j]*v1_i + y[j]*v2_i - p[j] <= x[i]*v1_i + y[i]*v2_i - p[i]
    #    => (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    
    A_rows = []
    b_rows = []
    
    # IR constraints
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[n_types + i] = -v2
        row[2*n_types + i] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    # IC constraints (skip i==j since they're trivially satisfied)
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[j] = v1_i
            row[i] = -v1_i
            row[n_types + j] = v2_i
            row[n_types + i] = -v2_i
            row[2*n_types + j] = -1
            row[2*n_types + i] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[i] <= 1, 0 <= y[i] <= 1, p[i] >= 0
    bounds = [(0, 1)] * n_types + [(0, 1)] * n_types + [(0, None)] * n_types
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a trivial feasible solution
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x[:n_types]
        y = result.x[n_types:2*n_types]
        p = result.x[2*n_types:3*n_types]
        
        # Clip to ensure feasibility
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        p = np.maximum(p, 0)
        
        # Verify IR with tolerance
        for i in range(n_types):
            v1, v2 = types[i]
            utility = x[i]*v1 + y[i]*v2 - p[i]
            if utility < -1e-8:
                p[i] = max(0.0, x[i]*v1 + y[i]*v2)
        
        # Verify IC with tolerance
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                util_report = x[j]*v1_i + y[j]*v2_i - p[j]
                util_truth = x[i]*v1_i + y[i]*v2_i - p[i]
                if util_report - util_truth > 1e-8:
                    # IC violated; fall back to trivial solution
                    allocations = np.zeros((n_types, 2))
                    payments = np.zeros(n_types)
                    return {'allocations': allocations, 'payments': payments}
        
        allocations = np.column_stack((x, y))
        return {'allocations': allocations, 'payments': p}
        
    except Exception:
        # Fallback: return a trivial feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
