import numpy as np
import time
import scipy.optimize as opt
import scipy.signal as sig

N = 600
TIME_LIMIT = 140
# Interval width
h = (0.5 / N)
# Grid centers
x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * h

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def smooth_max(conv, sigma=0.05):
    """
    Approximate the max of the autoconvolution using a soft-max (log-sum-exp) 
    or simply a Gaussian blur of the peaks. 
    For convexity in the basis coefficients, we can use the fact that 
    max(conv) is convex. 
    However, to make it differentiable for gradient descent, we can use 
    a smooth approximation. 
    Actually, the objective is convex. 
    Let's use scipy.optimize.minimize with BFGS on the basis coefficients.
    """
    # We want to minimize max(conv) / sum(f)^2
    # This is equivalent to minimizing max(conv) subject to sum(f)=1.
    # Or minimizing max(conv) with f normalized.
    
    # A simple smooth max approximation: 
    # log(sum(exp(beta * conv))) / beta
    # But conv can be negative? No, f>=0 so conv>=0.
    # However, beta needs to be large enough.
    # Alternatively, since the domain is discrete, we can just use the actual max 
    # and rely on sub-gradient or just use a differentiable surrogate.
    
    # Let's try a differentiable surrogate: 
    # max(x) approx = log(sum(exp(x/sigma))) * sigma
    # But this is expensive if N is large. 
    # N=600, conv size ~1200. 
    # exp(1200 values) is fine.
    
    # However, we need to be careful with scaling.
    # Let's just use the actual max in the objective and let scipy handle it 
    # if we use a method that doesn't require gradients? 
    # No, BFGS requires gradients. 
    # Let's implement the gradient of the max.
    # If we know which lag is the max, the gradient is the gradient of that lag.
    # If we don't, we can use a smooth approximation.
    
    # Let's use a smooth approximation with adaptive sigma.
    # Or better: use the "log-sum-exp" trick which is smooth and convex.
    
    # The objective is:
    # J(c) = max_m (c^T A_m c) / (c^T 1)^2
    # This is a ratio of convex functions? No.
    # But if we fix the normalization sum(f)=1, we minimize max(c^T A_m c).
    # This is a convex problem!
    # min_c max_m c^T A_m c
    # s.t. sum(c_i * w_i) = 1 (where w_i are weights of basis functions)
    # c_i >= 0
    
    # We can solve this using scipy.optimize.minimize with a smooth max approximation.
    
    beta = 50.0 # Sharpness of max approximation
    # To avoid overflow, subtract max
    c_max = np.max(conv)
    if c_max > 0:
        # log-sum-exp approximation of max
        # log(sum(exp(beta * (conv - c_max)))) / beta + c_max
        # But we want the value itself, not just the log.
        # Actually, just using np.max is non-smooth.
        # Let's use a Gaussian blur of the peaks to get a smooth proxy for the max.
        # Or simply: the max is one of the conv values.
        # For gradient descent, we can just pick the argmax and compute the gradient of that specific term.
        # This is a subgradient method.
        
        # Let's implement a helper to compute the value and gradient of the objective
        # with respect to the basis coefficients.
        pass

def basis_functions(params):
    """
    Generate the step function heights from basis parameters.
    params: [center_1, width_1, height_1, center_2, width_2, height_2, ...]
    We use a sum of Gaussians (or sincs) to ensure smoothness and non-negativity.
    """
    f = np.zeros(N)
    n_funcs = len(params) // 3
    for i in range(n_funcs):
        c = params[3*i]
        w = params[3*i+1]
        hgt = params[3*i+2]
        # Gaussian
        f += hgt * np.exp(-0.5 * ((x - c) / w) ** 2)
    return np.maximum(f, 0)

def objective_and_grad(params):
    """
    Compute the objective (normalized max autoconv) and its gradient w.r.t params.
    We use the subgradient of the max function.
    """
    f = basis_functions(params)
    s = np.sum(f)
    if s < 1e-10:
        return 1e10, np.zeros_like(params)
    
    # Normalize f to have sum 1 for the objective calculation
    # But we want to optimize the shape. 
    # The score is homogeneous, so we can just minimize max(conv) with sum(f)=1 constraint?
    # Or minimize max(conv) / s^2.
    # Let's minimize J = max(conv) / s^2.
    
    conv = fft_autoconv(f)
    max_val = np.max(conv)
    score = max_val / (s * s)
    
    # Gradient calculation:
    # d(score)/d(f_i) = [ (d(max)/d(f_i)) * s^2 - max_val * 2 * s * (1) ] / s^4
    # = [ d(max)/d(f_i) * s - 2 * max_val ] / s^3
    
    # d(max(conv))/d(f_i) is the gradient of the specific lag that achieved the max.
    # If multiple lags achieve the max, we sum their gradients (subgradient).
    
    # Let's find all lags where conv == max_val (within tolerance)
    tol = 1e-8
    max_lags = np.where(np.abs(conv - max_val) < tol)[0]
    
    grad_f = np.zeros(N)
    for m in max_lags:
        # conv[m] = sum_j f[j] f[m-j]
        # d(conv[m])/d(f[i]) = f[m-i] + f[i-m] (if indices valid)
        # Note: conv is symmetric.
        # For a given m, the terms involving f[i] are:
        # 1. j=i, m-j=i => m=2i. Term: 2 f[i] (derivative is 2 f[i]? No, d(f[i]^2)/df[i] = 2f[i])
        #    Wait, conv[m] = sum_{j=0}^{N-1} f[j] f[m-j].
        #    If m-j = i => j = m-i. Term f[m-i] f[i].
        #    If j = i => m-j = i => m=2i. Term f[i] f[i].
        #    
        #    Generally, d(conv[m])/d(f[i]) = f[m-i] + f[i-m]
        #    where f[k] is 0 if k is out of bounds [0, N-1].
        
        # Vectorized update for this lag m:
        # We need to add f[m-i] and f[i-m] to grad_f[i] for all i.
        # This is equivalent to adding f and a flipped f to the gradient.
        
        # Let's do it simply for the few max lags.
        for i in range(N):
            j1 = m - i
            j2 = i - m
            grad_val = 0.0
            if 0 <= j1 < N:
                grad_val += f[j1]
            if 0 <= j2 < N:
                grad_val += f[j2]
            grad_f[i] += grad_val / len(max_lags) # Average if multiple max lags
            
    # Now chain rule to params
    # f = basis_functions(params)
    # We need J_f = (grad_f * s - 2 * max_val * 1) / s**3  (derivative w.r.t f_i)
    # Wait, derivative of max_val w.r.t f_i is grad_f[i].
    # d(score)/df_i = (grad_f[i] * s^2 - max_val * 2s) / s^4 = (grad_f[i] * s - 2 max_val) / s^3
    
    J_f = (grad_f * s - 2 * max_val) / (s ** 3)
    
    # Now dJ/dparams = J_f @ df/dparams
    # df/dparams is a sparse matrix.
    # For each basis function k (center c_k, width w_k, height h_k):
    # f = sum h_k exp(-0.5 ((x-c_k)/w_k)^2)
    # df/dh_k = exp(...)
    # df/dc_k = h_k * exp(...) * (x-c_k)/w_k^2
    # df/dw_k = h_k * exp(...) * (x-c_k)^2 / w_k^3
    
    grad_params = np.zeros_like(params)
    n_funcs = len(params) // 3
    for k in range(n_funcs):
        c = params[3*k]
        w = params[3*k+1]
        h = params[3*k+2]
        
        if w < 1e-6:
            continue
            
        # Compute basis shape
        shape = np.exp(-0.5 * ((x - c) / w) ** 2)
        
        # Contribution to grad_params[3*k+2] (height)
        grad_params[3*k+2] += np.dot(J_f, shape)
        
        # Contribution to grad_params[3*k] (center)
        # d shape / d c = shape * (x-c)/w^2
        grad_c = shape * (x - c) / (w * w)
        grad_params[3*k] += np.dot(J_f, h * grad_c)
        
        # Contribution to grad_params[3*k+1] (width)
        # d shape / d w = shape * (x-c)^2 / w^3
        grad_w = shape * (x - c) ** 2 / (w ** 3)
        grad_params[3*k+1] += np.dot(J_f, h * grad_w)
        
    return score, grad_params

def polish_step_function(f_init, start_time, max_time=120.0):
    """
    Local refinement of the step function using projected gradient descent.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Compute gradient of score w.r.t f
        conv = fft_autoconv(f)
        max_val = np.max(conv)
        max_lags = np.where(np.abs(conv - max_val) < 1e-8)[0]
        
        grad_f = np.zeros(N)
        for m in max_lags:
            # d(conv[m])/d(f[i]) = f[m-i] + f[i-m]
            # We can compute this efficiently:
            # grad_f += roll(f, -m) + roll(f, m) ?
            # Careful with boundaries.
            # f is defined on 0..N-1.
            # conv[m] = sum_{j=0}^{N-1} f[j] f[m-j]
            # d/d f[i] sum_j f[j] f[m-j]
            # Term 1: j=i => f[i] f[m-i]. Deriv: f[m-i] (if m-i in range)
            # Term 2: m-j=i => j=m-i => f[m-i] f[i]. Deriv: f[i] (if m-i in range)
            # So if m=2i, deriv is 2 f[i]. Else f[m-i] + f[i-m].
            
            # Efficient way:
            # Create a vector g where g[i] = f[m-i] + f[i-m]
            # This is the convolution of f with a delta at m?
            # Actually, the gradient of the max lag m is the "cross-correlation" of f with itself at lag m.
            # Which is just the components of conv? No.
            # Let's just loop for N=600, it's fast enough for 1000 steps?
            # 600 * 1000 = 600k ops. Yes.
            
            for i in range(N):
                j1 = m - i
                j2 = i - m
                g = 0.0
                if 0 <= j1 < N: g += f[j1]
                if 0 <= j2 < N: g += f[j2]
                grad_f[i] += g / len(max_lags)
                
        # Objective derivative
        # d(score)/df_i = (grad_f[i] * s - 2 * max_val) / s^3
        J_f = (grad_f * s - 2 * max_val) / (s ** 3)
        
        # Gradient step
        f_new = f - lr * J_f
        
        # Project to non-negative and normalize
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new > 1e-15:
            f_new = f_new / s_new
        else:
            f_new = f / s
            
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return best_f

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Define a small number of basis functions (e.g., 3-5 Gaussians)
    # 2. Optimize their parameters using scipy.optimize.minimize (BFGS)
    # 3. Convert to step function
    # 4. Polish with step-function gradient descent
    
    # Initial guess: Single Gaussian centered at 0
    # params: [c, w, h]
    # We'll try a few different initial configurations
    
    best_f = None
    best_score = np.inf
    
    # Configuration 1: Single Gaussian
    init_params = np.array([0.0, 0.1, 1.0])
    
    # Configuration 2: Two Gaussians
    init_params2 = np.array([0.0, 0.1, 1.0, 0.1, 0.1, 1.0])
    
    # Configuration 3: Three Gaussians
    init_params3 = np.array([0.0, 0.1, 1.0, 0.05, 0.05, 1.0, -0.05, 0.05, 1.0])
    
    candidates_params = [init_params, init_params2, init_params3]
    
    for p_init in candidates_params:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        try:
            # Use BFGS
            res = opt.minimize(objective_and_grad, p_init, method='BFGS', jac=True,
                               options={'maxiter': 200, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                f_opt = basis_functions(res.x)
                f_opt = np.maximum(f_opt, 0.0)
                s = np.sum(f_opt)
                if s > 1e-15:
                    f_opt = f_opt / s
                    score = compute_score(f_opt)
                    if score < best_score:
                        best_score = score
                        best_f = f_opt
        except Exception as e:
            pass
            
    # If no basis optimization succeeded, fallback to uniform
    if best_f is None:
        best_f = np.ones(N) / N
        best_score = compute_score(best_f)
        
    # Polish the best candidate
    if time.time() - start_time < TIME_LIMIT - 10:
        polished = polish_step_function(best_f, start_time, TIME_LIMIT - 5)
        polished_score = compute_score(polished)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
            
    # Final safety check
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
