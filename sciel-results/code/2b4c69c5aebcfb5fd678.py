import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the type space
    values = [0, 1, 2, 3]
    num_types = 16
    
    # Generate all type combinations (v1, v2)
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
            
    # Extract joint probabilities
    probs = np.zeros(num_types)
    
    if isinstance(distribution_spec, np.ndarray) and distribution_spec.shape == (4, 4):
        probs = distribution_spec.flatten()
    elif isinstance(distribution_spec, dict):
        if 'joint' in distribution_spec:
            joint = distribution_spec['joint']
            if isinstance(joint, np.ndarray) and joint.shape == (4, 4):
                probs = joint.flatten()
            elif isinstance(joint, list) and len(joint) == 16:
                probs = np.array(joint)
        elif 'independent' in distribution_spec:
            ind = distribution_spec['independent']
            p1 = np.array(ind[0])
            p2 = np.array(ind[1])
            joint = np.outer(p1, p2)
            probs = joint.flatten()
        elif 'p1' in distribution_spec and 'p2' in distribution_spec:
            p1 = np.array(distribution_spec['p1'])
            p2 = np.array(distribution_spec['p2'])
            joint = np.outer(p1, p2)
            probs = joint.flatten()
    elif isinstance(distribution_spec, (list, tuple)) and len(distribution_spec) == 2:
        p1 = np.array(distribution_spec[0])
        p2 = np.array(distribution_spec[1])
        joint = np.outer(p1, p2)
        probs = joint.flatten()
    else:
        probs = np.ones(num_types) / num_types
        
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(num_types) / num_types

    # Variables:
    # x_{k,0}, x_{k,1} for k in 0..15 (32 vars)
    # u_k for k in 0..15 (16 vars)
    # Total: 48
    
    n_vars = 48
    x_offset = 0
    u_offset = 32
    
    # Objective: Maximize Revenue
    c = np.zeros(n_vars)
    for k in range(num_types):
        v1, v2 = types[k]
        p_k = probs[k]
        c[x_offset + 2*k + 0] = -p_k * v1
        c[x_offset + 2*k + 1] = -p_k * v2
        c[u_offset + k] = p_k

    # Constraints
    n_ic = num_types * (num_types - 1)
    n_ir = num_types
    n_pay = num_types
    n_constraints = n_ic + n_ir + n_pay
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR: -u_k <= 0
    for k in range(num_types):
        row = n_ic + k
        A_ub[row, u_offset + k] = -1.0
        
    # IC: -u_v + u_v' + (v - v') . x_v' <= 0
    idx = 0
    for v_idx in range(num_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(num_types):
            if v_idx == vp_idx: continue
            vp1, vp2 = types[vp_idx]
            row = idx
            idx += 1
            A_ub[row, u_offset + v_idx] = -1.0
            A_ub[row, u_offset + vp_idx] = 1.0
            A_ub[row, x_offset + 2*vp_idx + 0] = v1 - vp1
            A_ub[row, x_offset + 2*vp_idx + 1] = v2 - vp2
            
    # Payment Non-Negativity: u_k - v1*x_k1 - v2*x_k2 <= 0
    for k in range(num_types):
        row = n_ic + n_ir + k
        v1, v2 = types[k]
        A_ub[row, u_offset + k] = 1.0
        A_ub[row, x_offset + 2*k + 0] = -v1
        A_ub[row, x_offset + 2*k + 1] = -v2
        
    bounds = [(0.0, 1.0)] * 32 + [(0.0, None)] * 16
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
        
        z = res.x
        allocations = np.zeros((16, 2))
        utilities = np.zeros(16)
        for k in range(num_types):
            allocations[k, 0] = z[x_offset + 2*k + 0]
            allocations[k, 1] = z[x_offset + 2*k + 1]
            utilities[k] = z[u_offset + k]
            
        payments = np.zeros(16)
        for k in range(num_types):
            v1, v2 = types[k]
            payments[k] = v1 * allocations[k, 0] + v2 * allocations[k, 1] - utilities[k]
            
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify
        for k in range(num_types):
            if payments[k] < -1e-8 or utilities[k] < -1e-8:
                return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
                
        return {"allocations": allocations, "payments": payments}
    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
