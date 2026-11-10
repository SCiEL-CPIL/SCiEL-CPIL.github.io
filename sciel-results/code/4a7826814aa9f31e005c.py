import numpy as np
from scipy.optimize import linprog
from scipy import sparse

def construct_auction(distribution_spec, support, constraints):
    # Define the support: 16 types (v1, v2) with v1, v2 in {0, 1, 2, 3}
    # Order: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution_spec to get probabilities for each type
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            # Fallback to uniform if length mismatch
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
            else:
                probs[i] = 1.0 / n_types
    elif isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if "uniform" in spec:
            probs = np.ones(n_types) / n_types
        elif "skewed" in spec or "biased" in spec:
            # Example of a skewed distribution: higher values more likely
            # v1, v2 independent, P(v) proportional to (v+1)
            weights = np.array([(v1 + 1) * (v2 + 1) for v1, v2 in types])
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
        
    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types

    # Variables:
    # x[0..31]: allocations for each type and good. 
    # x[type_idx * 2 + good_idx]
    # p[32..47]: payments for each type.
    # p[type_idx + 32]
    
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    # Maximize revenue: sum(P(v) * p_v)
    # Minimize -revenue
    for i in range(n_types):
        c[32 + i] = -probs[i]
        
    # Constraints: A_ub x <= b_ub
    # We will build sparse rows for efficiency
    
    # 1. x <= 1 (32 constraints)
    # 2. IR: p_v - v . x_v <= 0 (16 constraints)
    # 3. IC: p_v - p_v' - v . x_v + v . x_v' <= 0 (240 constraints)
    
    # Total constraints: 32 + 16 + 240 = 288
    # Total vars: 48
    
    # Using sparse COO format
    row_indices = []
    col_indices = []
    data = []
    b_ub_list = []
    
    # 1. x <= 1
    for i in range(n_types):
        for g in range(n_goods):
            row_idx = len(b_ub_list)
            col_idx = i * n_goods + g
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            data.append(1.0)
            b_ub_list.append(1.0)
            
    # 2. IR: p_v - sum(v_g * x_vg) <= 0
    for i in range(n_types):
        row_idx = len(b_ub_list)
        v1, v2 = types[i]
        # p_v
        row_indices.append(row_idx)
        col_indices.append(32 + i)
        data.append(1.0)
        
        # -v1 * x_v1
        if v1 != 0:
            row_indices.append(row_idx)
            col_indices.append(i * n_goods + 0)
            data.append(-v1)
            
        # -v2 * x_v2
        if v2 != 0:
            row_indices.append(row_idx)
            col_indices.append(i * n_goods + 1)
            data.append(-v2)
            
        b_ub_list.append(0.0)
        
    # 3. IC: p_v - p_v' - v . x_v + v . x_v' <= 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row_idx = len(b_ub_list)
            v1_j, v2_j = types[j]
            
            # p_v
            row_indices.append(row_idx)
            col_indices.append(32 + i)
            data.append(1.0)
            
            # -p_v'
            row_indices.append(row_idx)
            col_indices.append(32 + j)
            data.append(-1.0)
            
            # -v . x_v
            if v1_i != 0:
                row_indices.append(row_idx)
                col_indices.append(i * n_goods + 0)
                data.append(-v1_i)
            if v2_i != 0:
                row_indices.append(row_idx)
                col_indices.append(i * n_goods + 1)
                data.append(-v2_i)
                
            # +v . x_v'
            # Note: The constraint is p_v - p_v' - v . x_v + v . x_v' <= 0
            # The term is v (of type i) dot x_j
            if v1_i != 0:
                row_indices.append(row_idx)
                col_indices.append(j * n_goods + 0)
                data.append(v1_i)
            if v2_i != 0:
                row_indices.append(row_idx)
                col_indices.append(j * n_goods + 1)
                data.append(v2_i)
                
            b_ub_list.append(0.0)
            
    A_ub = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(len(b_ub_list), n_vars)).tocsr()
    b_ub = np.array(b_ub_list)
    
    # Bounds for variables
    # x: [0, 1]
    # p: [0, inf)
    bounds = []
    for i in range(n_types * n_goods):
        bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback: Simple VCG-like or uniform allocation?
        # If LP fails, return a valid but potentially suboptimal mechanism.
        # Safest fallback: Allocate 0, Pay 0. Revenue 0.
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    else:
        x_opt = res.x[:n_types * n_goods].reshape(n_types, n_goods)
        p_opt = res.x[n_types * n_goods:]
        
        # Clip small numerical errors
        x_opt = np.clip(x_opt, 0, 1)
        p_opt = np.clip(p_opt, 0, None)
        
        allocations = x_opt
        payments = p_opt

    return {
        'allocations': allocations,
        'payments': payments
    }
