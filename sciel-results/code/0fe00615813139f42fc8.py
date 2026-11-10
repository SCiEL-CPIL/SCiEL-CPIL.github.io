import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def _parse_distribution(distribution_spec, n_types, types):
    """
    Parse distribution spec into a probability vector of length n_types.
    Handles:
    - None -> Uniform
    - dict with 'probs' -> Normalize and use
    - dict with 'distribution': 'uniform', 'geometric', 'independent'
    - list/tuple/ndarray -> Normalize and use
    """
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            try:
                probs = np.array(distribution_spec['probs'], dtype=float).flatten()
                if len(probs) == n_types:
                    probs = np.maximum(probs, 0)
                    s = probs.sum()
                    if s > 0:
                        return probs / s
                return np.ones(n_types) / n_types
            except Exception:
                return np.ones(n_types) / n_types
        
        if 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if dist == 'uniform':
                return np.ones(n_types) / n_types
            elif dist == 'geometric':
                # P(v1=k, v2=l) proportional to r^k * r^l, default r=0.7
                r = 0.7
                if 'r' in distribution_spec:
                    r = distribution_spec['r']
                probs = np.array([r**v1 * r**v2 for v1, v2 in types], dtype=float)
                probs = np.maximum(probs, 0)
                s = probs.sum()
                if s > 0:
                    return probs / s
                return np.ones(n_types) / n_types
            elif dist == 'independent':
                # Try to parse independent marginals p1, p2
                if 'p1' in distribution_spec and 'p2' in distribution_spec:
                    try:
                        p1 = np.array(distribution_spec['p1'], dtype=float)
                        p2 = np.array(distribution_spec['p2'], dtype=float)
                        if len(p1) == 4 and len(p2) == 4:
                            probs = np.zeros(n_types)
                            for idx, (v1, v2) in enumerate(types):
                                probs[idx] = p1[v1] * p2[v2]
                            probs = np.maximum(probs, 0)
                            s = probs.sum()
                            if s > 0:
                                return probs / s
                    except Exception:
                        pass
                return np.ones(n_types) / n_types
            else:
                return np.ones(n_types) / n_types
        else:
            return np.ones(n_types) / n_types
            
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == n_types:
                probs = np.maximum(probs, 0)
                s = probs.sum()
                if s > 0:
                    return probs / s
            return np.ones(n_types) / n_types
        except Exception:
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
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
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
    # linprog minimizes, so we negate the coefficients for p[i]
    c = np.zeros(n_vars)
    c[2*n_types:3*n_types] = -probs
    
    # Constraints:
    # 1. IR: For each type i, p[i] <= x[i]*v1_i + y[i]*v2_i
    #    => -x[i]*v1_i - y[i]*v2_i + p[i] <= 0
    # 2. IC: For each true type i and report j,
    #    u_i(j) <= u_i(i)
    #    x[j]*v1_i + y[j]*v2_i - p[j] <= x[i]*v1_i + y[i]*v2_i - p[i]
    #    (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    
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
    
    # IC constraints (skip i==j since they are 0<=0)
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # Coefficients for x
            row[j] = v1_i
            row[i] = -v1_i
            # Coefficients for y
            row[n_types + j] = v2_i
            row[n_types + i] = -v2_i
            # Coefficients for p
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
        # Use 'highs' for robustness and speed
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a trivial feasible solution (zero allocations, zero payments)
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x[:n_types]
        y = result.x[n_types:2*n_types]
        p = result.x[2*n_types:3*n_types]
        
        # Clip to ensure feasibility within bounds
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        p = np.maximum(p, 0)
        
        # Verify IR with tolerance
        # u_i = x[i]*v1 + y[i]*v2 - p[i] >= -1e-8
        for i in range(n_types):
            v1, v2 = types[i]
            utility = x[i]*v1 + y[i]*v2 - p[i]
            if utility < -1e-8:
                # If IR is violated, we must reduce payment to satisfy IR
                # This might break IC, but it's better than an invalid solution
                # However, the LP solution should satisfy IR. If not, it's a solver error.
                # We clamp p[i] to the max feasible value under IR.
                p[i] = max(0.0, x[i]*v1 + y[i]*v2)
        
        # Verify IC with tolerance
        # u_i(j) - u_i(i) <= 1e-8
        ic_violated = False
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                util_report = x[j]*v1_i + y[j]*v2_i - p[j]
                util_truth = x[i]*v1_i + y[i]*v2_i - p[i]
                if util_report - util_truth > 1e-8:
                    ic_violated = True
                    break
            if ic_violated:
                break
        
        if ic_violated:
            # If IC is violated, the solution is invalid.
            # For a maximization problem, the zero solution is always feasible.
            # Returning zero ensures validity, though revenue is 0.
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
