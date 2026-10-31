import numpy as np
import time
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    return np.max(conv) / (total * total)

def _smooth_max_obj_grad(h, beta):
    """
    Compute smooth max autoconvolution value and gradient w.r.t. h.
    Uses log-sum-exp with temperature 1/beta.
    """
    h = np.maximum(h, 1e-10)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    h_norm = h / total
    conv = _fft_convolve(h_norm)
    
    m = np.max(conv)
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_conv = np.exp(shifted)
    sum_exp = np.sum(exp_conv)
    weights = exp_conv / sum_exp
    
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h_norm via FFT
    size = 2 * N
    W = fft(weights, size)
    H = fft(h_norm, size)
    grad_norm = np.real(ifft(W * H, size))[:N]
    grad_norm *= 2.0
    
    # Chain rule for f(h) = soft_max(conv(h/S))
    sum_grad_u = np.dot(grad_norm, h_norm)
    grad_h = (grad_norm - sum_grad_u) / total
    
    return soft_max_val, grad_h

def _cosine_poly_to_h(params):
    """
    Convert cosine polynomial parameters to discrete heights.
    params: [width, amp0, cos1, cos2, ..., cosM]
    h(x) = amp0 * exp(-x^2/(2*width^2)) * (1 + sum_k cos_k * cos(k * pi * x / L))
    where x is in [-1/4, 1/4], L = 1/8 (half-width)
    """
    width = max(params[0], 0.05)
    amps = params[1:]  # cos coefficients
    i = np.arange(N)
    # Map indices to [-1/4, 1/4]
    x = -0.25 + 0.5 * i / (N - 1)
    
    # Gaussian envelope
    g = np.exp(-x**2 / (2 * width**2))
    
    # Cosine polynomial (even terms only for symmetry)
    poly = np.ones(N)
    for k in range(len(amps)):
        # Use odd harmonics to maintain even symmetry around center
        # Actually, let's use all harmonics but ensure the function is even
        # For even symmetry, we need cos(k * pi * x / L) where L is half-width
        # x ranges from -1/4 to 1/4, so x/L ranges from -2 to 2 when L=1/8
        # Let's use x in units of 1/4, so x_scaled = x / 0.25 = 4x, range [-1, 1]
        x_scaled = 4.0 * x  # range [-1, 1]
        poly += amps[k] * np.cos((k + 1) * np.pi * x_scaled / 2.0)
    
    h = g * poly
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 1e-10:
        h /= s
    else:
        h = np.ones(N) / N
    return h

def _gmm_to_h(params):
    """Convert GMM parameters to discrete heights."""
    K = 4
    h = np.zeros(N)
    i = np.arange(N)
    for k in range(K):
        w_k = max(params[3*k], 0.01)
        mu_k = params[3*k+1]
        sig_k = max(params[3*k+2], 1.0)
        h += w_k * np.exp(-((i - mu_k)**2) / (2 * sig_k**2))
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h /= s
    else:
        h = np.ones(N) / N
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()

    best_score = np.inf
    best_h = None
    
    # Strategy 1: GMM Initialization + Differential Evolution
    K = 4
    
    def gmm_objective(params):
        h = _gmm_to_h(params)
        return compute_autoconv_max(h)
    
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))
        bounds.append((0, N-1))
        bounds.append((5.0, N/3.0))
    
    # Initial guess: uniform
    best_h = np.ones(N) / N
    best_score = compute_autoconv_max(best_h)
    
    if time_left() > 50:
        try:
            max_iter_de = int(min(40, time_left() // 1.5))
            if max_iter_de < 10:
                max_iter_de = 10
            
            de_result = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds,
                seed=42,
                maxiter=max_iter_de,
                popsize=12,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_score:
                h_de = _gmm_to_h(de_result.x)
                best_score = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass
    
    # Strategy 2: Annealed L-BFGS-B on raw h
    if time_left() > 40 and best_h is not None:
        betas = [10.0, 50.0, 200.0, 500.0, 1000.0]
        current_h = best_h.copy()
        current_val = best_score
        
        for beta in betas:
            if time_left() < 5:
                break
            try:
                res = optimize.minimize(
                    lambda h: _smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 150}
                )
                if res.success:
                    final_h = np.maximum(res.x, 0)
                    if np.sum(final_h) > 0:
                        final_h /= np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            except Exception:
                pass
        
        if current_val < best_score:
            best_score = current_val
            best_h = current_h.copy()
    
    # Strategy 3: Cosine polynomial parameterization + gradient-based optimization
    # This is the key refinement: use a smooth parametric family for final polish
    if time_left() > 25 and best_h is not None:
        # Initialize cosine parameters from best_h
        # Use width ~ N/4, and small initial cosine coefficients
        M = 6  # Number of cosine terms
        init_params = np.zeros(M + 1)
        init_params[0] = N / 4.0  # width in index space
        
        # Estimate initial cosine coefficients via least squares
        # Project best_h onto the cosine basis
        i = np.arange(N)
        x_scaled = 4.0 * (-0.25 + 0.5 * i / (N - 1))  # range [-1, 1]
        
        # Build design matrix for cosine terms
        A = np.zeros((N, M))
        for k in range(M):
            A[:, k] = np.cos((k + 1) * np.pi * x_scaled / 2.0)
        
        # Weight by the Gaussian envelope to get better projection
        width_est = N / 4.0
        g_weights = np.exp(-(x_scaled**2) / (2 * (width_est / (4 * N * 0.25))**2))
        # Simplify: just use uniform weights for projection
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            init_params[1:] = coeffs
        except:
            init_params[1:] = 0.0
        
        def cosine_objective(params):
            h = _cosine_poly_to_h(params)
            return compute_autoconv_max(h)
        
        def cosine_smooth_obj_grad(params, beta):
            h = _cosine_poly_to_h(params)
            val, grad_h = _smooth_max_obj_grad(h, beta)
            # We need grad w.r.t. params, but for simplicity use finite differences
            # or just use the raw h optimization
            return val
        
        # Use L-BFGS-B on the parametric space with finite difference gradients
        # First, do a few rounds of direct h optimization, then switch to parametric
        current_params = init_params.copy()
        current_val = compute_autoconv_max(_cosine_poly_to_h(current_params))
        
        # Annealed optimization on parametric space
        betas_param = [20.0, 100.0, 500.0]
        for beta in betas_param:
            if time_left() < 5:
                break
            try:
                # Use finite difference gradient
                def param_obj(p):
                    h = _cosine_poly_to_h(p)
                    return _smooth_max_obj_grad(h, beta)[0]
                
                res = optimize.minimize(
                    param_obj,
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 100,
                             'eps': 1e-6}
                )
                if res.success:
                    final_h = _cosine_poly_to_h(res.x)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < current_val:
                        current_val = final_val
                        current_params = res.x.copy()
                        best_h = final_h.copy()
                        best_score = min(best_score, current_val)
            except Exception:
                pass
        
        # Final polish: switch back to raw h with very high beta
        if time_left() > 10 and best_h is not None:
            current_h = best_h.copy()
            current_val = best_score
            for beta in [1000.0, 2000.0]:
                if time_left() < 3:
                    break
                try:
                    res = optimize.minimize(
                        lambda h: _smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 80}
                    )
                    if res.success:
                        final_h = np.maximum(res.x, 0)
                        if np.sum(final_h) > 0:
                            final_h /= np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()
                except Exception:
                    pass
            
            if current_val < best_score:
                best_score = current_val
                best_h = current_h.copy()
    
    # Strategy 4: Multi-restart with random GMM + quick polish
    if time_left() > 15:
        n_restarts = int(min(3, time_left() // 8))
        for r in range(n_restarts):
            if time_left() < 6:
                break
            params = np.zeros(3*K)
            for k in range(K):
                params[3*k] = np.random.uniform(0.1, 2.0)
                params[3*k+1] = np.random.uniform(0, N)
                params[3*k+2] = np.random.uniform(10, N/4)
            
            h_init = _gmm_to_h(params)
            current_h = h_init
            current_val = compute_autoconv_max(current_h)
            
            try:
                for beta in [50.0, 200.0]:
                    if time_left() < 3:
                        break
                    res = optimize.minimize(
                        lambda h: _smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 60}
                    )
                    if res.success:
                        final_h = np.maximum(res.x, 0)
                        if np.sum(final_h) > 0:
                            final_h /= np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()
            except Exception:
                pass
            
            if current_val < best_score:
                best_score = current_val
                best_h = current_h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)
