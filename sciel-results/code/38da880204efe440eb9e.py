import time
import numpy as np

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalized score: for uniform heights=1, score = 1.0
    score = max_conv * n / (total * total * w)
    return score

def _autoconv_and_grad(heights, alpha=200.0):
    """Compute smooth max of autoconvolution and its gradient w.r.t. heights."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    m = np.max(conv)
    if m < 0:
        m = 0.0
    exp_conv = np.exp(alpha * (conv - m))
    weights = exp_conv / np.sum(exp_conv)
    
    sm = m + np.log(np.sum(exp_conv)) / alpha
    
    # Gradient of smooth_max w.r.t. f
    W_full = np.fft.rfft(weights)
    F_full = np.fft.rfft(f)
    corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
    grad_f = 2.0 * corr[:n]
    
    return sm, grad_f

def _smooth_obj_and_grad(heights, alpha=200.0):
    """Compute the smooth objective and its gradient."""
    sm, grad_f = _autoconv_and_grad(heights, alpha)
    
    S = np.sum(heights)
    w = 0.5 / N
    if S <= 1e-10:
        return 1e10, np.zeros(N)
    
    obj = sm * N / (S * S * w)
    obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
    
    return obj, obj_grad

def _mirror_descent_optimize(h_init, time_limit):
    """
    Optimize using Mirror Descent (multiplicative weights) for the non-negative
    simplex-like constraint. This handles non-negativity more naturally by
    working in log-space or using exponential updates.
    """
    start = time.time()
    deadline = start + time_limit
    
    def time_left():
        return deadline - time.time()
    
    h = h_init.copy()
    h = np.maximum(h, 1e-10)  # Ensure strictly positive for log
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Initialize step size
    lr = 0.1
    alpha = 200.0
    
    # We'll use a projected gradient in the log domain
    # Let y = log(h), so h = exp(y)
    # The objective in terms of y is:
    #   J(y) = smooth_max(conv(exp(y))) * N / (sum(exp(y))^2 * w)
    # 
    # Actually, let's use a simpler approach: multiplicative updates
    # h_new = h * exp(-lr * grad_log)
    # where grad_log = grad_obj / h (gradient w.r.t. log h)
    
    for it in range(5000):
        if time_left() < 1.0:
            break
        
        obj, grad = _smooth_obj_and_grad(h, alpha)
        
        # Gradient w.r.t. log h: grad_log_i = grad_i / h_i
        grad_log = grad / np.maximum(h, 1e-10)
        
        # Multiplicative update
        h_new = h * np.exp(-lr * grad_log)
        h_new = np.maximum(h_new, 1e-15)
        
        # Check if improvement
        score_new = compute_score(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
            # Slightly increase lr if improving
            lr = min(lr * 1.01, 0.5)
        else:
            # Decrease lr
            lr = max(lr * 0.9, 0.001)
            # Don't update h, keep current
        
        # Periodically increase alpha for sharper max approximation
        if it % 200 == 0 and it > 0:
            alpha = min(alpha * 1.1, 1000.0)
        
        # Early stopping if convergence
        if it > 100 and abs(score_new - best_score) < 1e-8:
            if it % 100 == 0:
                break
    
    return best_h, best_score

def _projected_gradient_optimize(h_init, time_limit):
    """
    Optimize using projected gradient descent with adaptive step size.
    """
    start = time.time()
    deadline = start + time_limit
    
    def time_left():
        return deadline - time.time()
    
    h = h_init.copy()
    h = np.maximum(h, 0.0)
    best_h = h.copy()
    best_score = compute_score(h)
    
    lr = 0.001
    alpha = 200.0
    momentum = np.zeros(N)
    beta = 0.9
    
    for it in range(5000):
        if time_left() < 1.0:
            break
        
        obj, grad = _smooth_obj_and_grad(h, alpha)
        
        # Momentum
        momentum = beta * momentum + (1 - beta) * grad
        h_new = h - lr * momentum
        h_new = np.maximum(h_new, 0.0)
        
        score_new = compute_score(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
            lr = min(lr * 1.02, 0.01)
        else:
            lr = max(lr * 0.85, 1e-5)
        
        if it % 200 == 0 and it > 0:
            alpha = min(alpha * 1.1, 1000.0)
    
    return best_h, best_score

def _refine_with_fewer_dims(h_init, time_limit):
    """
    Try to refine by assuming a parametric form with fewer parameters.
    """
    start = time.time()
    deadline = start + time_limit
    
    def time_left():
        return deadline - time.time()
    
    # Try a Gaussian-like shape: h(x) = a * exp(-b * (x - c)^2)
    from scipy.optimize import minimize
    
    def parametric_heights(params):
        a, b, c = params
        x = np.linspace(0, 1, N)
        h = a * np.exp(-b * (x - c) ** 2)
        return np.maximum(h, 0.0)
    
    def parametric_score(params):
        h = parametric_heights(params)
        if np.sum(h) <= 0:
            return 1e10
        return compute_score(h)
    
    # Initialize from current best
    h_curr = h_init
    S = np.sum(h_curr)
    if S <= 0:
        return h_init, compute_score(h_init)
    
    # Estimate initial parameters
    c_init = np.sum(h_curr * np.linspace(0, 1, N)) / S
    var = np.sum(h_curr * (np.linspace(0, 1, N) - c_init)**2) / S
    b_init = 1.0 / max(var, 1e-6)
    a_init = np.max(h_curr)
    
    best_params = np.array([a_init, b_init, c_init])
    best_score = compute_score(h_init)
    
    # Optimize
    for _ in range(3):
        if time_left() < 2.0:
            break
        res = minimize(parametric_score, best_params, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
        else:
            break
    
    h_opt = parametric_heights(best_params)
    score_opt = compute_score(h_opt)
    
    if score_opt < best_score:
        return h_opt, score_opt
    return h_init, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Phase 1: Mirror Descent optimization
    if time_left() > 60:
        h_md, score_md = _mirror_descent_optimize(h, time_left() - 5)
        if score_md < best_score:
            best_score = score_md
            best_h = h_md.copy()
    
    # Phase 2: Projected gradient refinement
    if time_left() > 30:
        h_pg, score_pg = _projected_gradient_optimize(best_h, time_left() - 5)
        if score_pg < best_score:
            best_score = score_pg
            best_h = h_pg.copy()
    
    # Phase 3: Parametric refinement
    if time_left() > 15:
        h_ref, score_ref = _refine_with_fewer_dims(best_h, time_left() - 3)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    # Phase 4: Final fine-tuning with small steps
    if time_left() > 10:
        h = best_h.copy()
        lr = 0.0001
        alpha = 500.0
        for it in range(1000):
            if time_left() < 2.0:
                break
            obj, grad = _smooth_obj_and_grad(h, alpha)
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0.0)
            score_new = compute_score(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                h = h_new.copy()
            else:
                lr = max(lr * 0.9, 1e-6)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
