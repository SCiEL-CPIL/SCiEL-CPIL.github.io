import numpy as np
from scipy.optimize import minimize
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s

# Precompute FFT setup
# The step function has N intervals on [-1/4, 1/4], total width 0.5
# We use a fine grid for accurate convolution
# Each step interval gets K fine samples
K = 32
M_STEP = N * K  # 19200
M_FFT = 1 << 15  # 32768 (power of 2, >= M_STEP)

# Precompute the index mapping for fine grid
def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < 0):
        return np.inf
    
    # Create fine-sampled function
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # The circular convolution is correct due to zero-padding
    # Find the maximum
    max_conv = np.max(conv)
    
    # Compute L2 norm squared (integral of f^2)
    # integral = sum(f(x)^2 * dx), dx = 0.5/M_STEP
    dx = 0.5 / M_STEP
    norm2 = np.sum(f_fine**2) * dx
    
    if norm2 < 1e-12:
        return np.inf
    
    # Score = max_conv / norm2
    # Note: the "normalized maximum" is ||f*f||_inf / ||f||_2^2
    score = max_conv / norm2
    
    return score

def compute_score_and_grad(heights):
    """Compute score and gradient using soft-max for smoothness."""
    heights = np.asarray(heights, dtype=np.float64)
    
    # Create fine-sampled function
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    # Compute autoconvolution via FFT
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Soft-max for smooth peak approximation
    beta = 100.0
    c_max = np.max(conv)
    if c_max < 1e-12:
        c_max = 1e-12
    
    exp_terms = np.exp(beta * (conv - c_max))
    Z = np.sum(exp_terms)
    soft_max = (1.0 / beta) * np.log(Z)
    
    # L2 norm squared
    dx = 0.5 / M_STEP
    norm2 = np.sum(f_fine**2) * dx
    
    if norm2 < 1e-12:
        return 1e10, np.zeros(N)
    
    # Objective: soft_max / norm2
    # We want to minimize this
    obj = soft_max / norm2
    
    # Gradient computation
    # d(soft_max)/d(conv) = exp_terms / Z
    # d(conv)/d(f_fine) = 2 * f_fine (for autoconvolution, derivative is 2*f convolved, but for linear conv it's f shifted)
    # Actually, for autoconvolution c = f * f, dc/df = 2 * f * (where * is convolution)
    # In frequency: C = F^2, dC/dF = 2F, so dc/df = ifft(2F * ifft(dC/df_fine))
    # More precisely: c[n] = sum_k f[k] f[n-k]
    # dc/d f[m] = sum_k (delta(n-m-k) f[n-k] + f[k] delta(n-k-m)) = 2 * f convolved with delta shifted
    # In FFT: dC/df_fine corresponds to ifft(2*F * ifft(W)) where W = d(soft_max)/d(conv)
    
    # Weight vector
    W = exp_terms / Z  # d(soft_max)/d(conv)
    
    # Gradient of soft_max w.r.t. f_fine
    # conv = ifft(fft(f_fine)^2)
    # d(conv)/d(f_fine) in frequency domain: if we perturb f_fine by delta, 
    # delta(conv) = ifft(2 * F * fft(delta))
    # So gradient contribution = ifft(2 * F * fft(W))
    
    W_fft = np.fft.rfft(W)
    grad_f_fine = np.fft.irfft(2 * F * W_fft, n=M_FFT)
    
    # Gradient of norm2 w.r.t. f_fine: 2 * f_fine * dx
    # d(obj)/d(f_fine) = (d(soft_max)/d(f_fine) * norm2 - soft_max * d(norm2)/d(f_fine)) / norm2^2
    # = (grad_f_fine * norm2 - soft_max * 2 * f_fine * dx) / norm2^2
    
    grad_f = (grad_f_fine * norm2 - soft_max * 2.0 * f_fine * dx) / (norm2 ** 2)
    
    # Map gradient from fine grid to step heights
    # Each height h[i] affects K consecutive fine samples
    # grad_h[i] = sum of grad_f over the K samples for interval i
    grad_h = np.zeros(N, dtype=np.float64)
    grad_h = np.sum(grad_f[:M_STEP].reshape(N, K), axis=1)
    
    return obj, grad_h

def construct():
    global START_TIME
    
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Initial guess: raised cosine
    x = np.linspace(0, 1, N)
    h0 = 0.5 * (1 + np.cos(2 * np.pi * x))
    h0 = np.maximum(h0, 0.01)  # Small floor to avoid zero
    
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    # Optimization 1: L-BFGS-B with gradient on soft-max surrogate
    try:
        def obj_and_grad(h):
            h = np.maximum(h, 0)  # Clip to nonnegative
            obj, grad = compute_score_and_grad(h)
            # Zero out gradient where h is at boundary (0)
            grad = np.where(h < 1e-10, 0.0, grad)
            return obj, grad
        
        res = minimize(
            obj_and_grad,
            h0,
            method='L-BFGS-B',
            jac=True,
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'maxfun': 500, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if hasattr(res, 'x') and res.x is not None:
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
                
    except Exception:
        pass
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 15:
        return [max(0.0, float(v)) for v in best_h]
    
    # Optimization 2: Different starting point - uniform
    try:
        h1 = np.ones(N)
        
        def obj_and_grad2(h):
            h = np.maximum(h, 0)
            obj, grad = compute_score_and_grad(h)
            grad = np.where(h < 1e-10, 0.0, grad)
            return obj, grad
        
        res2 = minimize(
            obj_and_grad2,
            h1,
            method='L-BFGS-B',
            jac=True,
            bounds=[(0, None)] * N,
            options={'maxiter': 80, 'maxfun': 400, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if hasattr(res2, 'x') and res2.x is not None:
            h_opt2 = np.maximum(res2.x, 0)
            score_opt2 = compute_score(h_opt2)
            if score_opt2 < best_score:
                best_h = h_opt2
                best_score = score_opt2
                
    except Exception:
        pass
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return [max(0.0, float(v)) for v in best_h]
    
    # Optimization 3: Different starting point - Gaussian-like
    try:
        xg = np.linspace(-1, 1, N)
        h2 = np.exp(-2 * xg**2)
        h2 = h2 / np.max(h2)
        
        def obj_and_grad3(h):
            h = np.maximum(h, 0)
            obj, grad = compute_score_and_grad(h)
            grad = np.where(h < 1e-10, 0.0, grad)
            return obj, grad
        
        res3 = minimize(
            obj_and_grad3,
            h2,
            method='L-BFGS-B',
            jac=True,
            bounds=[(0, None)] * N,
            options={'maxiter': 80, 'maxfun': 400, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if hasattr(res3, 'x') and res3.x is not None:
            h_opt3 = np.maximum(res3.x, 0)
            score_opt3 = compute_score(h_opt3)
            if score_opt3 < best_score:
                best_h = h_opt3
                best_score = score_opt3
                
    except Exception:
        pass
    
    # Final polish with tighter tolerances if time allows
    if time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            def obj_and_grad_polish(h):
                h = np.maximum(h, 0)
                obj, grad = compute_score_and_grad(h)
                grad = np.where(h < 1e-10, 0.0, grad)
                return obj, grad
            
            res_polish = minimize(
                obj_and_grad_polish,
                best_h,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'maxfun': 200, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if hasattr(res_polish, 'x') and res_polish.x is not None:
                h_polish = np.maximum(res_polish.x, 0)
                score_polish = compute_score(h_polish)
                if score_polish < best_score:
                    best_h = h_polish
                    best_score = score_polish
                    
        except Exception:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return [max(0.0, float(v)) for v in best_h]
