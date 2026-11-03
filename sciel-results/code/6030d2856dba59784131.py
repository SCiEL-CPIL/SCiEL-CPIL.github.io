import numpy as np
import scipy.optimize as opt
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Variables: h (length N) and M (scalar)
    # x = [h_0, ..., h_{N-1}, M]
    
    def autoconv(h):
        # Linear convolution of h with h
        # Length 2N - 1
        return np.convolve(h, h)

    def neg_ratio_obj(x):
        # We are minimizing M
        return x[N]

    def neg_ratio_obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    # Constraints:
    # 1. sum(h) == 1
    # 2. M - (h*h)_k >= 0 for all k
    
    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
        # We want M - c_k >= 0
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        # Gradient of (M - c_k) wrt x
        # d/dh_i (M - c_k) = - d/dh_i c_k
        # d/dh_i c_k = 2 * h[k-i] if valid index, else 0
        # Note: c_k = sum_j h_j h_{k-j}
        # d c_k / d h_i = h_{k-i} + h_{k-i} = 2 h_{k-i} (with h_out_of_bounds=0)
        
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            # grad[k, :N] = -2 * h[k - i] for i in 0..N-1
            # This is equivalent to -2 * reverse(h) shifted
            # Let's just loop or use slicing
            # h[k-i] is non-zero if 0 <= k-i < N
            # i ranges from max(0, k-(N-1)) to min(N-1, k)
            
            # Efficient way:
            # The vector of derivatives wrt h for a fixed k is:
            # d_k[i] = 2 * h[k-i]
            # This is 2 * h reversed and shifted?
            # Let's just compute it simply as N is small (50)
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0 # d/dM (M - c_k) = 1
            
        return grad

    # Initial guess
    # Uniform distribution
    h_init = np.ones(N) / N
    c_init = autoconv(h_init)
    M_init = np.max(c_init)
    x0 = np.concatenate([h_init, [M_init]])
    
    # Bounds
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    # Constraints dict
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    best_h = h_init
    best_ratio = -np.inf
    
    def calculate_ratio(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = autoconv(h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (l1 * peak)
    
    current_ratio = calculate_ratio(h_init)
    if current_ratio > best_ratio:
        best_ratio = current_ratio
        best_h = h_init.copy()
        
    try:
        # Use SLSQP
        res = opt.minimize(
            neg_ratio_obj, 
            x0, 
            method='SLSQP', 
            jac=neg_ratio_obj_grad,
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if res.success:
            h_res = res.x[:N]
            # Normalize to ensure sum is exactly 1
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                h_res = best_h
            # Clip negative values due to numerical errors
            h_res = np.clip(h_res, 0, None)
            h_res = h_res / np.sum(h_res)
            
            ratio_res = calculate_ratio(h_res)
            if ratio_res > best_ratio:
                best_ratio = ratio_res
                best_h = h_res.copy()
                
    except Exception as e:
        pass
    
    # Try a few random sparse restarts if time allows
    # This helps escape local optima if SLSQP got stuck
    while time.time() < deadline - 5.0:
        # Random sparse support
        K = np.random.randint(2, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        c_rand = autoconv(h_rand)
        M_rand = np.max(c_rand)
        x_rand = np.concatenate([h_rand, [M_rand]])
        
        try:
            res = opt.minimize(
                neg_ratio_obj, 
                x_rand, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
        except Exception:
            continue

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
