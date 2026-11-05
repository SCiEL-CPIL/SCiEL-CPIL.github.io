"""Incentive-compatible finite auction design via LP over allocation probabilities."""
import time
import numpy as np
from scipy.optimize import linprog, linprog  # linprog is the key solver

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Parameters:
    - distribution_spec: string or dict specifying the distribution over types
    - support: the set of types (values in {0,1,2,3} for each of 2 goods)
    - constraints: additional constraints (ignored in this implementation)
    
    Returns:
    - dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = 60.0  # seconds, safety margin before 90s kill
    
    # Define the type space: 16 types, each (v1, v2) with v1, v2 in {0,1,2,3}
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Determine the distribution
    if isinstance(distribution_spec, str):
        spec_lower = distribution_spec.lower()
        if 'uniform' in spec_lower or spec_lower == 'independent_uniform':
            probs = np.ones(n_types) / n_types
        elif 'uniform' in spec_lower:
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        for t, p in distribution_spec.items():
            if t in type_to_idx:
                probs[type_to_idx[t]] = p
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs is valid
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (n_types,):
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    total = probs.sum()
    if total < 1e-12:
        probs = np.ones(n_types) / n_types
    else:
        probs = probs / total
    
    # The mechanism is defined by allocation probabilities x[i, j] for each type i
    # where j is the bundle index. Since there are 2 goods, the possible bundles are:
    # 0: empty set, 1: {good 1}, 2: {good 2}, 3: {good 1, good 2}
    # But wait, the output is allocations of shape (16, 2), so each type gets a 2-vector
    # representing the allocation of each good (probability of allocating each good).
    # So x[i] is a 2-vector in [0,1]^2.
    
    # Let x[i, 0] = probability of allocating good 1 to type i
    # Let x[i, 1] = probability of allocating good 2 to type i
    
    # The expected utility of type i when reporting type j is:
    # U(i, j) = x[j, 0] * v_i[0] + x[j, 1] * v_i[1] - p[j]
    
    # Incentive compatibility: U(i, i) >= U(i, j) for all i, j
    # Individual rationality: U(i, i) >= 0 for all i
    
    # By Myerson's lemma, for a 1-dimensional type space, IC is equivalent to:
    # 1. x is non-decreasing
    # 2. payment identity: p(i) = v_i * x(i) - integral_0^v x(t) dt
    
    # But here we have a 2-dimensional type space with discrete types.
    # For multi-dimensional types, IC is more complex.
    
    # Let's formulate this as an LP.
    # Variables: x[i, 0], x[i, 1] for i in 0..15, and p[i] for i in 0..15
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    
    # Constraints:
    # 1. 0 <= x[i, g] <= 1 for all i, g
    # 2. p[i] >= 0 for all i
    # 3. IC: x[j,0]*v_i[0] + x[j,1]*v_i[1] - p[j] <= x[i,0]*v_i[0] + x[i,1]*v_i[1] - p[i] for all i, j
    #    Rearranged: p[i] - p[j] <= x[i,0]*v_i[0] + x[i,1]*v_i[1] - x[j,0]*v_i[0] - x[j,1]*v_i[1]
    #    Wait, let me be more careful.
    #    U(i, i) >= U(i, j)
    #    x[i,0]*v_i[0] + x[i,1]*v_i[1] - p[i] >= x[j,0]*v_i[0] + x[j,1]*v_i[1] - p[j]
    #    p[i] - p[j] <= x[i,0]*v_i[0] + x[i,1]*v_i[1] - x[j,0]*v_i[0] - x[j,1]*v_i[1]
    # 4. IR: x[i,0]*v_i[0] + x[i,1]*v_i[1] - p[i] >= 0 for all i
    #    p[i] <= x[i,0]*v_i[0] + x[i,1]*v_i[1]
    
    # Let's build the LP.
    # Variable ordering: [x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], p[1], ..., p[15]]
    # Index for x[i,g] = 2*i + g
    # Index for p[i] = 2*n_types + i
    
    n_vars = 2 * n_types + n_types  # 48
    
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog, we minimize, so we negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[2 * n_types + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: p[i] - x[i,0]*v_i[0] - x[i,1]*v_i[1] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v_i = types[i]
        row[2 * i + 0] = -v_i[0]
        row[2 * i + 1] = -v_i[1]
        row[2 * n_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # IC constraints: for all i, j (i != j):
    # p[i] - p[j] - x[i,0]*v_i[0] - x[i,1]*v_i[1] + x[j,0]*v_i[0] + x[j,1]*v_i[1] <= 0
    # Wait, let me re-derive.
    # U(i,i) >= U(i,j)
    # x[i,0]*v_i[0] + x[i,1]*v_i[1] - p[i] >= x[j,0]*v_i[0] + x[j,1]*v_i[1] - p[j]
    # -p[i] + p[j] >= x[j,0]*v_i[0] + x[j,1]*v_i[1] - x[i,0]*v_i[0] - x[i,1]*v_i[1]
    # p[i] - p[j] <= x[i,0]*v_i[0] + x[i,1]*v_i[1] - x[j,0]*v_i[0] - x[j,1]*v_i[1]
    # Rearranged for A_ub @ x <= b_ub:
    # p[i] - p[j] - x[i,0]*v_i[0] - x[i,1]*v_i[1] + x[j,0]*v_i[0] + x[j,1]*v_i[1] <= 0
    
    for i in range(n_types):
        v_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # p[i] coefficient: +1
            row[2 * n_types + i] = 1
            # p[j] coefficient: -1
            row[2 * n_types + j] = -1
            # x[i,0] coefficient: -v_i[0]
            row[2 * i + 0] = -v_i[0]
            # x[i,1] coefficient: -v_i[1]
            row[2 * i + 1] = -v_i[1]
            # x[j,0] coefficient: +v_i[0]
            row[2 * j + 0] = v_i[0]
            # x[j,1] coefficient: +v_i[1]
            row[2 * j + 1] = v_i[1]
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    # Bounds: 0 <= x[i,g] <= 1, 0 <= p[i] <= infinity
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x[i,0]
        bounds.append((0, 1))  # x[i,1]
    for i in range(n_types):
        bounds.append((0, None))  # p[i]
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_sol = result.x[:2 * n_types].reshape(n_types, 2)
            p_sol = result.x[2 * n_types:]
            
            # Clip to ensure feasibility
            x_sol = np.clip(x_sol, 0, 1)
            p_sol = np.clip(p_sol, 0, None)
            
            return {
                'allocations': x_sol,
                'payments': p_sol
            }
        else:
            # Fallback: return a feasible but possibly suboptimal mechanism
            # Use the trivial mechanism: allocate nothing, charge nothing
            x_sol = np.zeros((n_types, 2))
            p_sol = np.zeros(n_types)
            return {
                'allocations': x_sol,
                'payments': p_sol
            }
    except Exception as e:
        # Fallback
        x_sol = np.zeros((n_types, 2))
        p_sol = np.zeros(n_types)
        return {
            'allocations': x_sol,
            'payments': p_sol
        }
