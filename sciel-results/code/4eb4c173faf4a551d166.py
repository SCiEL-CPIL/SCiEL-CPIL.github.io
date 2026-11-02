import numpy as np
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -1.0
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # The ratio is scale-invariant in shape. We optimize the shape v (sum to 1)
    # and the ratio = ||v||^2 / (1 * ||v*v||_inf) since ||v||_1 = 1
    # Actually: ratio = ||v||^2 / (||v||_1 * max_conv(v)) = ||v||^2 / max_conv(v) when ||v||_1=1
    
    # Use soft-max for smooth approximation of max in autoconvolution
    # soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
    # This is differentiable and approximates max as beta -> inf
    
    def objective(z, beta=50.0):
        """
        z = log(v) where v are the shape weights (nonnegative)
        We want to maximize ||v||^2 / (||v||_1 * soft_max(conv(v), beta))
        Equivalently minimize the negative.
        
        To handle the constraint that v >= 0, we work in log-space: v = exp(z)
        The ratio is scale-invariant, so we can normalize ||v||_1 = 1.
        """
        v = np.exp(z)
        # Normalize so ||v||_1 = 1
        s = np.sum(v)
        if s <= 0:
            return 0.0
        v = v / s
        
        l2_sq = np.dot(v, v)
        conv = np.convolve(v, v, mode='full')
        
        # Soft max approximation of max(conv)
        # Use a stable log-sum-exp
        beta_local = beta
        max_conv = np.max(conv)
        if max_conv <= 0:
            return 0.0
        # log-sum-exp for stability
        shifted = beta_local * (conv - max_conv)
        log_sum_exp = max_conv + np.log(np.sum(np.exp(shifted))) / beta_local
        
        ratio = l2_sq / log_sum_exp
        if ratio <= 0:
            return 0.0
        return -ratio
    
    def objective_grad(z, beta=50.0):
        """Gradient of the objective (negative ratio) with respect to z."""
        v = np.exp(z)
        s = np.sum(v)
        if s <= 0:
            return np.zeros(N)
        v = v / s
        
        l2_sq = np.dot(v, v)
        conv = np.convolve(v, v, mode='full')
        
        beta_local = beta
        max_conv = np.max(conv)
        if max_conv <= 0:
            return np.zeros(N)
        
        shifted = beta_local * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_conv + np.log(sum_exp) / beta_local
        
        ratio = l2_sq / log_sum_exp
        if ratio <= 0:
            return np.zeros(N)
        
        # Gradient computation (numerical for safety, but let's try analytical)
        # Actually, let's use numerical gradient for robustness
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            fp = objective(z_plus, beta)
            fm = objective(z_minus, beta)
            grad[i] = (fp - fm) / (2 * eps)
        return grad
    
    # Better: use a parameterization that naturally keeps things positive and normalized
    # Use softmax parameterization: v_i = exp(w_i) / sum(exp(w_j))
    # This ensures v > 0 and ||v||_1 = 1 automatically
    
    def objective_softmax(w, beta=50.0):
        """w is the parameter, v = softmax(w)"""
        # Stable softmax
        w_max = np.max(w)
        v = np.exp(w - w_max)
        v = v / np.sum(v)
        
        l2_sq = np.dot(v, v)
        conv = np.convolve(v, v, mode='full')
        
        max_conv = np.max(conv)
        if max_conv <= 0:
            return 0.0
        
        # Soft max
        shifted = beta * (conv - max_conv)
        log_sum_exp = max_conv + np.log(np.sum(np.exp(shifted))) / beta
        
        ratio = l2_sq / log_sum_exp
        if ratio <= 0:
            return 0.0
        return -ratio
    
    # Generate diverse starting points in w-space
    x = np.linspace(-1, 1, N)
    
    start_points = []
    
    # Triangular
    t = 1 - np.abs(x)
    t = np.maximum(t, 1e-10)
    start_points.append(np.log(t))
    
    # Various Gaussians
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        g = np.exp(-x**2 / (2 * sigma**2))
        g = np.maximum(g, 1e-10)
        start_points.append(np.log(g))
    
    # Uniform
    start_points.append(np.zeros(N))
    
    # Narrow triangles
    for width in [0.2, 0.3, 0.4, 0.5]:
        t = np.maximum(0, 1 - np.abs(x) / width)
        t = np.maximum(t, 1e-10)
        start_points.append(np.log(t))
    
    # Bimodal
    for sep in [0.3, 0.5, 0.7]:
        b = np.exp(-(x - sep)**2 / 0.1) + np.exp(-(x + sep)**2 / 0.1)
        b = np.maximum(b, 1e-10)
        start_points.append(np.log(b))
    
    best_score = 0.0
    best_v = None
    
    # Optimize from each starting point
    for i, w0 in enumerate(start_points):
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Use L-BFGS-B for efficiency
            res = minimize(objective_softmax, w0, method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
            
            w_opt = res.x
            # Compute the actual shape
            w_max = np.max(w_opt)
            v_opt = np.exp(w_opt - w_max)
            v_opt = v_opt / np.sum(v_opt)
            
            score = compute_ratio(v_opt)
            
            if score > best_score:
                best_score = score
                best_v = v_opt.copy()
        except Exception:
            continue
    
    # Final polish with higher precision and more iterations
    if time.time() < deadline - 3.0 and best_v is not None:
        try:
            w_best = np.log(np.maximum(best_v, 1e-15))
            # Use higher beta for sharper max approximation
            res = minimize(objective_softmax, w_best, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12})
            
            w_final = res.x
            w_max = np.max(w_final)
            v_final = np.exp(w_final - w_max)
            v_final = v_final / np.sum(v_final)
            
            score_final = compute_ratio(v_final)
            if score_final > best_score:
                best_v = v_final
                best_score = score_final
        except Exception:
            pass
    
    # Additional refinement: try even sharper soft-max
    if time.time() < deadline - 2.0 and best_v is not None:
        try:
            w_best = np.log(np.maximum(best_v, 1e-15))
            # Use a wrapper with higher beta
            def obj_high_beta(w):
                return objective_softmax(w, beta=100.0)
            
            res = minimize(obj_high_beta, w_best, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            
            w_final = res.x
            w_max = np.max(w_final)
            v_final = np.exp(w_final - w_max)
            v_final = v_final / np.sum(v_final)
            
            score_final = compute_ratio(v_final)
            if score_final > best_score:
                best_v = v_final
                best_score = score_final
        except Exception:
            pass
    
    # Scale the result. Since the ratio is scale-invariant in shape,
    # we can use any positive scale. Use 1e-6 for numerical safety.
    if best_v is not None:
        scale = 1e-6
        h_out = best_v * scale
        h_out = np.maximum(h_out, 0.0)
        return h_out.tolist()
    else:
        # Fallback: uniform
        h_out = np.ones(N) * 1e-6
        return h_out.tolist()
