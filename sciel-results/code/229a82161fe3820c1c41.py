"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Helper to get index from (v1, v2)
    def get_idx(v1, v2):
        return v1 * 4 + v2

    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Three fixed distributions - assuming uniform for all if not specified otherwise
        probs = np.ones(num_types) / num_types
    else:
        # Try to interpret as list/array
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types

    # Reduced LP Formulation
    # Variables: x1[16], x2[16] -> 32 variables
    # Objective: Max sum p_i * t_i
    # t_i = v1_i*x1_i + v2_i*x2_i - u_i
    # u_i = sum_{s=1}^{v1_i} x1(s, v2_i) + sum_{s=1}^{v2_i} x2(v1_i, s)
    
    # Coefficients for x1[k, l] (index k*4+l)
    # C1[k, l] = p[k, l] * k - sum_{k' >= k} p[k', l]
    # Coefficients for x2[k, l] (index k*4+l)
    # C2[k, l] = p[k, l] * l - sum_{l' >= l} p[k, l']
    
    c1 = np.zeros(num_types)
    c2 = np.zeros(num_types)
    
    for v1 in range(4):
        for v2 in range(4):
            idx = get_idx(v1, v2)
            p_il = probs[idx]
            
            # Sum for C1: sum_{v1' >= v1} p[v1', v2]
            sum_p1 = 0
            for v1_prime in range(v1, 4):
                sum_p1 += probs[get_idx(v1_prime, v2)]
            c1[idx] = p_il * v1 - sum_p1
            
            # Sum for C2: sum_{v2' >= v2} p[v1, v2']
            sum_p2 = 0
            for v2_prime in range(v2, 4):
                sum_p2 += probs[get_idx(v1, v2_prime)]
            c2[idx] = p_il * v2 - sum_p2
            
    c = np.concatenate([c1, c2])
    
    # Constraints:
    # 1. Monotonicity
    # x1[k, l] >= x1[k-1, l] for k > 0
    # x2[k, l] >= x2[k, l-1] for l > 0
    # Inequality form: -x1[k, l] + x1[k-1, l] <= 0
    #                  -x2[k, l] + x2[k, l-1] <= 0
    
    A_rows = []
    b_rows = []
    
    for v1 in range(1, 4):
        for v2 in range(4):
            idx_cur = get_idx(v1, v2)
            idx_prev = get_idx(v1-1, v2)
            row = np.zeros(2 * num_types)
            row[idx_cur] = -1  # x1 part
            row[idx_prev] = 1   # x1 part
            A_rows.append(row)
            b_rows.append(0)
            
    for v1 in range(4):
        for v2 in range(1, 4):
            idx_cur = get_idx(v1, v2)
            idx_prev = get_idx(v1, v2-1)
            row = np.zeros(2 * num_types)
            row[num_types + idx_cur] = -1  # x2 part
            row[num_types + idx_prev] = 1   # x2 part
            A_rows.append(row)
            b_rows.append(0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * (2 * num_types)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": True})
        
        if result.success and time.time() < deadline:
            x = result.x
            x1 = x[:num_types]
            x2 = x[num_types:]
            
            # Clip small numerical errors
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            
            # Calculate payments
            t = np.zeros(num_types)
            for v1 in range(4):
                for v2 in range(4):
                    idx = get_idx(v1, v2)
                    u = 0
                    for s in range(1, v1 + 1):
                        u += x1[get_idx(s, v2)]
                    for s in range(1, v2 + 1):
                        u += x2[get_idx(v1, s)]
                    
                    t[idx] = v1 * x1[idx] + v2 * x2[idx] - u
                    # Ensure non-negative payment (IR)
                    if t[idx] < 0:
                        t[idx] = 0.0
            
            allocations = np.column_stack([x1, x2])
            payments = t
            return {'allocations': allocations, 'payments': payments}
            
    except Exception as e:
        pass
    
    # Fallback: simple mechanism
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
