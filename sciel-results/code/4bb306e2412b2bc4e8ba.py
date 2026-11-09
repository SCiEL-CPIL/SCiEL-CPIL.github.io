import numpy as np
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute CW masks
    cw_mask = np.zeros((D, n), dtype=np.int8)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
            
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # B: Load if all demands are routed CW
    B = (weights[:, None] * cw_mask).sum(axis=0)
    
    # D_vec[d]: Difference in load vector if demand d is switched from CW to CCW
    ones = np.ones(n, dtype=np.int8)
    D_vec = weights[:, None] * (ones[None, :] - 2 * cw_mask)
    
    # Exhaustive search (fast for 2^15)
    B_list = B.tolist()
    D_vec_list = D_vec.tolist()
    
    best_max_load = float('inf')
    best_gray = 0
    current_sum = [0.0] * n
    prev_gray = 0
    
    for g in range(2**D):
        gray = g ^ (g >> 1)
        
        if g > 0:
            diff = gray ^ prev_gray
            idx = diff.bit_length() - 1
            
            if (gray >> idx) & 1:
                dv = D_vec_list[idx]
                for e in range(n):
                    current_sum[e] += dv[e]
            else:
                dv = D_vec_list[idx]
                for e in range(n):
                    current_sum[e] -= dv[e]
                
        prev_gray = gray
        
        max_load = 0.0
        for e in range(n):
            load = B_list[e] + current_sum[e]
            if load > max_load:
                max_load = load
                
        if max_load < best_max_load:
            best_max_load = max_load
            best_gray = gray
    
    # Now try to improve with continuous optimization
    # Define objective: smooth approximation of max load
    def objective(x):
        # x is a vector of length D, values in [0,1]
        # Load on edge e = B[e] + sum_d x[d] * D_vec[d][e]
        # Use log-sum-exp for smooth max
        loads = B + x @ D_vec
        alpha = 50.0
        # Soft max using log-sum-exp
        max_val = np.max(loads)
        return alpha * (max_val + np.log(np.sum(np.exp(alpha * (loads - max_val)))))
    
    # Gradient
    def gradient(x):
        loads = B + x @ D_vec
        alpha = 50.0
        max_val = np.max(loads)
        exp_vals = np.exp(alpha * (loads - max_val))
        softmax = exp_vals / np.sum(exp_vals)
        # d/dx_d objective = alpha * sum_e softmax[e] * D_vec[d][e]
        return alpha * (softmax @ D_vec)
    
    # Try multiple starting points
    best_continuous_max = best_max_load
    best_continuous_solution = None
    
    # Start from the exhaustive solution
    x0 = np.zeros(D)
    for d in range(D):
        if (best_gray >> d) & 1:
            x0[d] = 1.0
            
    # Also try random starts
    np.random.seed(42)
    starts = [x0]
    for _ in range(20):
        starts.append(np.random.rand(D))
    
    bounds = [(0, 1)] * D
    
    for x_start in starts:
        try:
            result = minimize(objective, x_start, method='L-BFGS-B', 
                            jac=gradient, bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-10})
            if result.success:
                # Evaluate the actual max load at this continuous solution
                x_sol = result.x
                loads = B + x_sol @ D_vec
                actual_max = np.max(loads)
                
                # Round to binary and evaluate
                x_rounded = (x_sol > 0.5).astype(float)
                loads_rounded = B + x_rounded @ D_vec
                actual_max_rounded = np.max(loads_rounded)
                
                if actual_max_rounded < best_continuous_max:
                    best_continuous_max = actual_max_rounded
                    best_continuous_solution = x_rounded
        except Exception:
            pass
    
    # Also try a few more polish steps around the best continuous solution
    if best_continuous_solution is not None:
        # Try all binary solutions within Hamming distance 1 of the best
        for d in range(D):
            x_try = best_continuous_solution.copy()
            x_try[d] = 1.0 - x_try[d]
            loads = B + x_try @ D_vec
            max_load = np.max(loads)
            if max_load < best_continuous_max:
                best_continuous_max = max_load
                best_continuous_solution = x_try
        
        # Try Hamming distance 2
        for d1 in range(D):
            for d2 in range(d1+1, D):
                x_try = best_continuous_solution.copy()
                x_try[d1] = 1.0 - x_try[d1]
                x_try[d2] = 1.0 - x_try[d2]
                loads = B + x_try @ D_vec
                max_load = np.max(loads)
                if max_load < best_continuous_max:
                    best_continuous_max = max_load
                    best_continuous_solution = x_try
    
    # Use the better of exhaustive and continuous
    if best_continuous_solution is not None and best_continuous_max < best_max_load:
        return best_continuous_solution
    else:
        routing = np.zeros(D, dtype=float)
        for d in range(D):
            if (best_gray >> d) & 1:
                routing[d] = 1.0
            else:
                routing[d] = 0.0
        return routing
