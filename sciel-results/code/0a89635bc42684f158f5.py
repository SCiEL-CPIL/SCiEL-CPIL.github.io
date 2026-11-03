import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score_raw(h):
    """
    Compute the ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    # The ratio is scale invariant, so we can normalize h to sum=1
    # But we compute the raw ratio to be safe
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    
    if l_inf == 0:
        return 0.0
        
    ratio = l2_sq / (l1 * l_inf)
    return ratio

def _soft_max_obj_and_grad(h, beta):
    """
    Compute the soft-max approximation of the peak autoconvolution
    and its gradient.
    """
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Soft max
    # Shift to avoid overflow in exp
    max_val = np.max(conv)
    exp_conv = np.exp(beta * (conv - max_val))
    sum_exp = np.sum(exp_conv)
    
    # Objective value (soft max)
    obj = max_val + np.log(sum_exp) / beta
    
    # Weights for the gradient
    if sum_exp == 0:
        weights = np.zeros_like(conv)
    else:
        weights = exp_conv / sum_exp
        
    # Gradient of soft max is sum of weights * gradient of each term
    # grad q_k = 2 * h_reversed_shifted
    # We need to compute sum_k weights[k] * grad(q_k)
    
    # grad(q_k)_m = 2 * h_{k-m} if 0 <= k-m < N else 0
    # This is equivalent to 2 * (h * h)_m? No.
    # Let's compute it efficiently.
    # grad_obj = sum_k weights[k] * 2 * h_{k-m}
    # This is 2 * (weights * h)_m? No, weights is on conv indices (0..2N-2).
    # Let w be the weights array.
    # grad_m = 2 * sum_{k} w_k h_{k-m}
    # This is a convolution of w and h?
    # w is length 2N-1. h is length N.
    # conv(w, h, 'full')?
    # Let's check indices.
    # (w * h)_m = sum_k w_k h_{m-k}.
    # We want sum_k w_k h_{k-m}.
    # Let j = k-m => k = j+m.
    # sum_j w_{j+m} h_j.
    # This is cross-correlation or reversed convolution.
    # np.convolve(h, w, 'full')?
    # Let's just do it with a loop or vectorized shift if N is small. N=50 is small.
    
    grad = np.zeros(N)
    # For each m in 0..N-1
    # grad[m] = 2 * sum_{k} weights[k] * h[k-m]
    # Valid k: 0 <= k < 2N-1 and 0 <= k-m < N => m <= k < m+N
    for m in range(N):
        k_start = m
        k_end = m + N
        # k must be within 0..2N-2
        k_start = max(0, k_start)
        k_end = min(len(weights), k_end)
        if k_start < k_end:
            # h indices are k - m, ranging from k_start-m to k_end-1-m
            h_slice = h[k_start - m : k_end - 1 - m]
            w_slice = weights[k_start : k_end]
            grad[m] = 2.0 * np.dot(w_slice, h_slice)
            
    return obj, grad

def _objective_and_grad(x, beta):
    h = np.maximum(x, 0) # SLSQP might go slightly negative, clip for eval? 
    # Better to use bounds in SLSQP.
    # But objective function should be smooth.
    # If we clip, gradient is 0 where h<0.
    # Let's assume SLSQP respects bounds.
    
    obj, grad = _soft_max_obj_and_grad(h, beta)
    return obj, grad

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Initial guesses
    inits = [
        np.ones(N) / N,
        np.zeros(N), # Delta
        np.zeros(N), # Bi-modal
        np.random.rand(N)
    ]
    inits[1][0] = 1.0
    inits[2][0] = 0.7
    inits[2][1] = 0.3
    inits[3] = inits[3] / np.sum(inits[3])
    
    for h_init in inits:
        h_init = np.maximum(h_init, 0)
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
            
    # Optimization
    # We want to minimize the peak autoconvolution.
    # This should maximize the ratio if the L2 norm doesn't drop too much.
    # Actually, minimizing peak conv is the primary driver for the "inequality" constant.
    
    # Use SLSQP
    # Constraints: sum(h) = 1
    # Bounds: 0 <= h_i <= 1
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Try different beta values
    betas = [10.0, 50.0, 100.0, 200.0]
    
    np.random.seed(42)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Try multiple starts
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Random start
            h_start = np.random.rand(N)
            if np.random.rand() < 0.3:
                # Spike start
                idx = np.random.randint(N)
                h_start = np.zeros(N)
                h_start[idx] = 1.0
                # Add small noise to avoid symmetry
                h_start += np.random.rand(N) * 0.1
            h_start = h_start / np.sum(h_start)
            
            # Define local objective for SLSQP
            def fun(x):
                return _objective_and_grad(x, beta)[0]
            
            def jac(x):
                return _objective_and_grad(x, beta)[1]
                
            try:
                res = minimize(fun, h_start, method='SLSQP', jac=jac, 
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 100, 'ftol': 1e-10})
                
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                h_opt = h_opt / np.sum(h_opt)
                
                s = _score_raw(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
                    
            except Exception:
                pass
                
    # Final check
    if time.time() < deadline - 5:
        # Try a few more random restarts with high beta
        for _ in range(10):
             if time.time() > deadline:
                 break
             h_start = np.random.rand(N)
             h_start = h_start / np.sum(h_start)
             try:
                 def fun(x):
                     return _objective_and_grad(x, 200.0)[0]
                 def jac(x):
                     return _objective_and_grad(x, 200.0)[1]
                 res = minimize(fun, h_start, method='SLSQP', jac=jac, 
                                bounds=bounds, constraints=constraints,
                                options={'maxiter': 50, 'ftol': 1e-10})
                 h_opt = np.maximum(res.x, 0)
                 h_opt = h_opt / np.sum(h_opt)
                 s = _score_raw(h_opt)
                 if s > best_score:
                     best_score = s
                     best_h = h_opt.copy()
             except Exception:
                 pass

    return list(best_h)
