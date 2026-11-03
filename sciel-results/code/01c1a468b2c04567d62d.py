import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _objective_fixed_k(h, k_star):
    """
    Returns (-score, gradient) for objective assuming k_star is argmax of conv.
    Objective to minimize: - L2^2 / (L1 * C_k)
    """
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0, np.zeros_like(h)
    
    l2_sq = np.sum(h**2)
    
    # C_k = sum_{i} h_i h_{k-i}
    # Efficient calculation
    # h_conv_k is element k of np.convolve(h, h, 'full')
    # But we need gradient too.
    
    # Calculate C
    # k_star is index in full convolution [0, 2N-2]
    # h_i * h_j with i+j = k_star
    # i ranges from max(0, k_star-(N-1)) to min(N-1, k_star)
    i_min = max(0, k_star - (N - 1))
    i_max = min(N - 1, k_star)
    
    if i_min > i_max:
        C = 0.0
    else:
        indices = np.arange(i_min, i_max + 1)
        partner_indices = k_star - indices
        C = np.sum(h[indices] * h[partner_indices])
        
    if C <= 0:
        return 0.0, np.zeros_like(h)
        
    score = l2_sq / (l1 * C)
    
    # Gradient of score w.r.t h
    # f = A / (B C)
    # df/dh = ( (2h) * (B C) - A * (1*C + B * dC/dh) ) / (B C)^2
    # dC/dh_j = 2 * h[k_star - j] if valid, else 0
    
    dC_dh = np.zeros(N)
    # j ranges 0..N-1
    # k_star - j must be in 0..N-1
    # j >= k_star - (N-1) and j <= k_star
    j_min = max(0, k_star - (N - 1))
    j_max = min(N - 1, k_star)
    if j_min <= j_max:
        js = np.arange(j_min, j_max + 1)
        partners = k_star - js
        dC_dh[js] = 2.0 * h[partners]
        
    grad_num = 2.0 * h * (l1 * C) - l2_sq * (C + l1 * dC_dh)
    grad_denom = (l1 * C)**2
    grad_score = grad_num / grad_denom
    
    # We want to minimize -score, so gradient is -grad_score
    return -score, -grad_score

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Gaussians
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 0:
                inits.append(h_g / np.sum(h_g))
                
    # Triangular
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                inits.append(h_t / np.sum(h_t))
                
    # Random Sparse
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        inits.append(h_s / np.sum(h_s))
        
    # Sidon-like: try to pick indices with distinct sums
    # Simple heuristic: pick indices that are far apart
    for _ in range(10):
        h_s = np.zeros(N)
        # Pick K indices
        K = 10
        # Start with random
        idx = np.random.choice(N, K, replace=False)
        # Sort
        idx = np.sort(idx)
        h_s[idx] = 1.0
        inits.append(h_s / np.sum(h_s))

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)},
        {'type': 'ineq', 'fun': lambda h: h, 'jac': lambda h: np.eye(N)}
    ]
    bounds = [(0.0, 1.0)] * N

    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        h_curr = h_init.copy()
        # Local optimization loop
        for _ in range(10):
            if time.time() > deadline - 5:
                break
                
            h_conv = np.convolve(h_curr, h_curr, mode='full')
            k_star = np.argmax(h_conv)
            
            def obj_func(h):
                val, _ = _objective_fixed_k(h, k_star)
                return val
                
            def grad_func(h):
                _, grad = _objective_fixed_k(h, k_star)
                return grad
                
            try:
                res = minimize(obj_func, h_curr, jac=grad_func, method='SLSQP', 
                              bounds=bounds, constraints=constraints, 
                              options={'maxiter': 100, 'ftol': 1e-12})
                if res.success:
                    h_new = np.maximum(0, res.x)
                    if np.sum(h_new) > 0:
                        h_new /= np.sum(h_new)
                        s_new = _score_raw(h_new)
                        if s_new > best_score:
                            best_score = s_new
                            best_h = h_new.copy()
                        # Check if k_star changed
                        h_conv_new = np.convolve(h_new, h_new, mode='full')
                        k_star_new = np.argmax(h_conv_new)
                        if k_star_new == k_star and np.linalg.norm(h_new - h_curr) < 1e-6:
                            break
                        h_curr = h_new
                    else:
                        break
                else:
                    break
            except Exception:
                break

    return list(best_h)
