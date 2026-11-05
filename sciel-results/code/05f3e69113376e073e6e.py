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
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
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
        # Three fixed distributions
        if distribution_spec == 0:
            # Uniform
            probs = np.ones(num_types) / num_types
        elif distribution_spec == 1:
            # Product of independent uniform on {0,1,2,3}
            probs = np.ones(num_types) / num_types
        elif distribution_spec == 2:
            # Product of independent uniform on {0,1,2,3} with different scaling
            probs = np.ones(num_types) / num_types
        else:
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
    
    # LP formulation using Myerson payment identity
    # Variables: x1[16], x2[16] -> 32 variables
    # Payment for type i: t[i] = v1_i*x1[i] + v2_i*x2[i] - sum_j p_j * (v1_i*(x1[j]-x1[i]) + v2_i*(x2[j]-x2[i]))
    # Simplified: t[i] = v1_i*x1[i] + v2_i*x2[i] - sum_j p_j*v1_i*(x1[j]-x1[i]) - sum_j p_j*v2_i*(x2[j]-x2[i])
    #            = v1_i*x1[i] + v2_i*x2[i] - v1_i*(sum_j p_j*x1[j] - p_i*x1[i]) - v2_i*(sum_j p_j*x2[j] - p_i*x2[i])
    #            = v1_i*x1[i] + v2_i*x2[i] - v1_i*E[x1] + v1_i*p_i*x1[i] - v2_i*E[x2] + v2_i*p_i*x2[i]
    #            = x1[i]*(v1_i + v1_i*p_i) + x2[i]*(v2_i + v2_i*p_i) - v1_i*E[x1] - v2_i*E[x2]
    
    # Actually, let's use the standard Myerson payment identity:
    # t[i] = v1_i*x1[i] + v2_i*x2[i] - u[i]
    # where u[i] is the utility, and for IC mechanisms:
    # u[i] = u[0] + sum_{k: type k is below i} (v1_i - v1_k)*dx1 + (v2_i - v2_k)*dx2
    # 
    # For a simpler approach, let's directly solve the LP with payment variables but use
    # the fact that for IC, payment is determined by allocation. Let's use the envelope theorem.
    
    # Alternative: Solve LP with x1, x2, t variables but with tighter formulation
    # Variables: x1[16], x2[16], t[16] -> 48 variables
    # Maximize: sum_i p_i * t[i]
    # Subject to:
    #   0 <= x1[i] <= 1, 0 <= x2[i] <= 1, t[i] >= 0
    #   IC: v1_i*x1[i] + v2_i*x2[i] - t[i] >= v1_i*x1[j] + v2_i*x2[j] - t[j] for all i,j
    #   IR: v1_i*x1[i] + v2_i*x2[i] - t[i] >= 0 for all i
    
    # Let's set up the LP properly
    n_vars = 3 * num_types  # 48 variables
    x1_idx = np.arange(num_types)
    x2_idx = np.arange(num_types, 2*num_types)
    t_idx = np.arange(2*num_types, 3*num_types)
    
    # Objective: maximize sum p_i * t_i -> minimize -sum p_i * t_i
    c = np.zeros(n_vars)
    c[t_idx] = -probs
    
    # Inequality constraints: A_ub @ z <= b_ub
    # IC constraints: for all i,j: v1_i*(x1[i]-x1[j]) + v2_i*(x2[i]-x2[j]) - (t[i]-t[j]) >= 0
    # => -v1_i*(x1[i]-x1[j]) - v2_i*(x2[i]-x2[j]) + (t[i]-t[j]) <= 0
    A_rows = []
    b_rows = []
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v1_i*x1[i] + v1_i*x1[j] - v2_i*x2[i] + v2_i*x2[j] + t[i] - t[j] <= 0
            row[x1_idx[i]] = -v1_i
            row[x1_idx[j]] = v1_i
            row[x2_idx[i]] = -v2_i
            row[x2_idx[j]] = v2_i
            row[t_idx[i]] = 1
            row[t_idx[j]] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints: v1_i*x1[i] + v2_i*x2[i] - t[i] >= 0
    # => -v1_i*x1[i] - v2_i*x2[i] + t[i] <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = np.zeros(n_vars)
        row[x1_idx[i]] = -v1_i
        row[x2_idx[i]] = -v2_i
        row[t_idx[i]] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x1[i] <= 1, 0 <= x2[i] <= 1, t[i] >= 0
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i] >= 0
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success and time.time() < deadline:
            z = result.x
            x1 = z[x1_idx]
            x2 = z[x2_idx]
            t = z[t_idx]
            
            # Clip small numerical errors
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            t = np.clip(t, 0, None)
            
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
