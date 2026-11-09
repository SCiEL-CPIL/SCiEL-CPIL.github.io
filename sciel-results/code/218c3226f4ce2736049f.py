import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Robust parsing of support
    # support could be list of lists, tuple, np.array
    support_arr = np.array(support, dtype=float)
    if support_arr.ndim == 1:
        if support_arr.size == 32:
            support_arr = support_arr.reshape(16, 2)
        elif support_arr.size == 16:
            # Maybe only one good? But task says two goods.
            # Assume it's malformed or needs reshaping.
            pass
    # Ensure shape (16, 2)
    if support_arr.shape != (16, 2):
        # Try to flatten and reshape
        try:
            support_arr = support_arr.flatten().reshape(16, 2)
        except:
            # Fallback to zeros if parsing fails
            support_arr = np.zeros((16, 2))
            
    # Robust parsing of distribution_spec
    pi_arr = np.array(distribution_spec, dtype=float).flatten()
    if pi_arr.size != 16:
        # Fallback
        pi_arr = np.ones(16)
    pi_arr = pi_arr / pi_arr.sum()
    
    n_reports = 16
    n_goods = 2
    
    def idx(r, t):
        return r * 3 + t
        
    n_vars = n_reports * 3
    
    # Objective: Minimize -Sum(pi_i * p_i)
    c = np.zeros(n_vars)
    for i in range(n_reports):
        v1, v2 = support_arr[i]
        c[idx(i, 0)] = -pi_arr[i] * v1
        c[idx(i, 1)] = -pi_arr[i] * v2
        c[idx(i, 2)] = pi_arr[i]
        
    A_rows = []
    b_rows = []
    
    # 1. IC Constraints
    for i in range(n_reports):
        for j in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            v_i1, v_i2 = support_arr[i]
            v_j1, v_j2 = support_arr[j]
            
            # (v_i1 - v_j1) x_j1 + (v_i2 - v_j2) x_j2 - u_i + u_j <= 0
            row[idx(j, 0)] = v_i1 - v_j1
            row[idx(j, 1)] = v_i2 - v_j2
            row[idx(i, 2)] = -1
            row[idx(j, 2)] = 1
            
            A_rows.append(row)
            b_rows.append(0)
            
    # 2. Non-negative payments: p_i >= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        v1, v2 = support_arr[i]
        row[idx(i, 0)] = -v1
        row[idx(i, 1)] = -v2
        row[idx(i, 2)] = 1
        
        A_rows.append(row)
        b_rows.append(0)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 100))
        
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
        
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(n_reports):
        allocations[i, 0] = res.x[idx(i, 0)]
        allocations[i, 1] = res.x[idx(i, 1)]
        u_i = res.x[idx(i, 2)]
        v1, v2 = support_arr[i]
        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - u_i
        
    payments[payments < 0] = 0
    allocations[allocations < 0] = 0
    allocations[allocations > 1] = 1
    
    return {"allocations": allocations, "payments": payments}
