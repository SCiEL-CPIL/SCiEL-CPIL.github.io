<<<<<<< SEARCH
for k in range(2, K + 1):
    v = float(mu[k])
    if v != 0.0:
        g[k::k] += v
=======
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    start_time = time.time()
    
    K = 200
    n_vars = K + 1
    
    c = np.zeros(n_vars)
    for i in range(1, n_vars):
        k = i + 1
        c[i] = np.log(k) / k
        
    A_eq = np.zeros((1, n_vars))
    for i in range(n_vars):
        k = i + 1
        A_eq[0, i] = 1.0 / k
    b_eq = [0.0]
    
    num_constraints = 10 * K
    A_ub = np.zeros((num_constraints, n_vars))
    b_ub = np.ones(num_constraints)
    
    for x_idx in range(num_constraints):
        x = x_idx + 1
        for i in range(n_vars):
            k = i + 1
            A_ub[x_idx, i] = x // k
            
    bounds = [(-10.0, 10.0)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            domain = np.arange(1, K + 1)
            values = res.x
            return domain, values
    except Exception:
        pass
        
    domain = np.arange(1, K + 1)
    values = np.zeros(K + 1)
    return domain, values
>>>>>>> REPLACE
