import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
import time

def construct():
    n = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_3aps = len(ap3s)
    num_4aps = len(ap4s)
    
    # Variables: x (n), z (num_3aps)
    total_vars = n + num_3aps
    
    # Objective: Maximize sum(z) -> Minimize -sum(z)
    c = np.zeros(total_vars)
    for k in range(num_3aps):
        c[n + k] = -1.0
        
    # Constraints
    rows = []
    cols = []
    vals = []
    b_list = []
    
    # 4-AP Constraints: sum(x_i) <= 3
    for ap in ap4s:
        for i in ap:
            rows.append(len(b_list))
            cols.append(i)
            vals.append(1.0)
        b_list.append(3.0)
        
    # Linking Constraints
    for k, (a, b, c_elem) in enumerate(ap3s):
        z_var = n + k
        
        # z - x_a <= 0
        rows.append(len(b_list))
        cols.append(z_var)
        vals.append(1.0)
        rows.append(len(b_list))
        cols.append(a)
        vals.append(-1.0)
        b_list.append(0.0)
        
        # z - x_b <= 0
        rows.append(len(b_list))
        cols.append(z_var)
        vals.append(1.0)
        rows.append(len(b_list))
        cols.append(b)
        vals.append(-1.0)
        b_list.append(0.0)
        
        # z - x_c <= 0
        rows.append(len(b_list))
        cols.append(z_var)
        vals.append(1.0)
        rows.append(len(b_list))
        cols.append(c_elem)
        vals.append(-1.0)
        b_list.append(0.0)
        
        # -z + x_a + x_b + x_c <= 2
        rows.append(len(b_list))
        cols.append(z_var)
        vals.append(-1.0)
        rows.append(len(b_list))
        cols.append(a)
        vals.append(1.0)
        rows.append(len(b_list))
        cols.append(b)
        vals.append(1.0)
        rows.append(len(b_list))
        cols.append(c_elem)
        vals.append(1.0)
        b_list.append(2.0)
        
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_list), total_vars))
    b = np.array(b_list)
    bounds = [(0, 1)] * total_vars
    
    start_time = time.time()
    
    x_sol = None
    
    # Try milp
    if hasattr(opt, 'milp'):
        try:
            res = opt.milp(
                c=c,
                constraints=A,
                integrality=np.array([1]*n + [0]*num_3aps),
                bounds=bounds,
                options={'time_limit': 55.0}
            )
            if res.x is not None:
                x_sol = res.x[:n]
        except Exception:
            pass
            
    # Fallback to linprog if milp failed or not available
    if x_sol is None:
        try:
            res = opt.linprog(
                c=c,
                A_ub=A,
                b_ub=b,
                bounds=bounds,
                method='highs'
            )
            if res.success and res.x is not None:
                # Round
                x_sol = np.round(res.x[:n])
        except Exception:
            pass
            
    if x_sol is None:
        return [1]
        
    state = x_sol.astype(int)
    
    # Repair
    changed = True
    while changed:
        changed = False
        for ap in ap4s:
            if all(state[i] == 1 for i in ap):
                state[ap[0]] = 0
                changed = True
                
    result = [i + 1 for i in range(n) if state[i] == 1]
    return result
